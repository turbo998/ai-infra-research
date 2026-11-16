# PyTorch 张量切片视图（`Tensor View / Slice`）底层 `UntypedStorage` 引用计数钉死显存泄漏与 `lm_head` 前置切片 `.clone()` 选型：为什么缓存一个仅 `128 KB` 的末尾隐状态切片 `H[:, -1, :]` 会把整块 `1.00 GB` 的序列显存死死钉在 HBM 里不释放（放大 `8,192 倍`，32 步狂漏 `32 GB`），而写 `lm_head(H)[:, -1, :]` 又会白白多算 `40.80 ms`？基于 `lm_head` 前置末位收集（`logits_to_keep=1`）+ 跨生命周期 `.clone()` 存储剥离（显存从 `32 GB` 降至 `4 MB`，Prefill 输出头提速 `510x`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/pytorch_tensor_view_slice_untyped_storage_pinning_memory_leak_clone_selection/run.py` 与 `results.json`。

![PyTorch Tensor View UntypedStorage Pinning Leak & Pre-lm_head Clone Visual Summary](../../assets/svg/pytorch-tensor-view-slice-untyped-storage-pinning-memory-leak-clone-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向大模型 Prefill 推理、Embedding/Reward 池化与自定义反向传播算子（`Transformers logits_to_keep=1 / vLLM Last-Token Gather / PyTorch saved_tensors_hooks`）的“`lm_head` 前置末位 Token 抽取（避免实例化 `9.96 GB` 全序列 Logits，提速 `510.0x`）+ 跨作用域张量视图 `.clone()` 物理存储解绑（`UntypedStorage` Detachment）+ `untyped_storage().nbytes() / t.nbytes()` 自动化泄漏审计门控”**。
- **架构师价值**：在开发大模型 RL 采样器、Embedding 向量提取服务、投机解码隐藏态缓存（如 `EAGLE-3` 缓存最后一层 `hidden_states`）或自定义 `torch.autograd.Function` 时，无数资深工程师都栽倒在过一行看似天经地义的 PyTorch 切片代码上：
  ```python
  # ⛔ 隐藏着 8,192 倍显存泄漏与 510 倍算力浪费的两行“教科书毒代码”！
  logits = self.lm_head(hidden_states)[:, -1, :]   # 陷阱 A: 先算满 8,192 个词的词表投影再切片!
  cached_last_h = hidden_states[:, -1, :]          # 陷阱 B: 128 KB 的切片把 1.00 GB 物理显存钉死了!
  del hidden_states                                # 以为删掉原变量就释放了 1 GB? 完全没有!
  ```
  为什么明明 `del hidden_states` 已经执行了，而且你用 `cached_last_h.numel() * cached_last_h.element_size()` 算出来的大小只有区区 **`128 KB`**（`[B=16, D=4096]` in `BF16`），但缓存了 32 个 Step 之后，80GB 的 H100 却突然爆掉 **`32.00 GB` 显存** 当场报 `CUDA out of memory`？
  1. **底层真相一：PyTorch 的 `Tensor` 只是轻量级视图头（`TensorImpl`），真正的显存挂在共享引用计数的 `UntypedStorage` 上（放大 `8,192 倍`，Tier 1 & Tier 2）**！
     - 在 PyTorch 底层 C++ 架构（`c10::TensorImpl` 与 `c10::StorageImpl`）中，张量切片操作 `hidden_states[:, -1, :]`、`x[:k]`、`narrow()`、`view()` **全部是零拷贝视图操作（View Operations）**！
     - 它们只是新建了一个修改了 `sizes=[16, 4096]`、`strides=[33554432, 1]` 和 `storage_offset` 的壳子，**而壳子底下的 `storage_ptr` 指向的依然是原来那个 `[16, 8192, 4096]`（整整 `1.00 GB`！）的底层物理显存块 `UntypedStorage`，并把该 `UntypedStorage` 的引用计数加了 `1`**！
     - 当你执行 `del hidden_states` 时，`1.00 GB` 物理显存块的引用计数从 `2` 减到了 `1`（因为 `cached_last_h` 还在引用它！）。**只要这个 `128 KB` 的小切片还活在列表、字典、LRU 缓存或 `ctx.save_for_backward` 里，那整块 `1.00 GB` 的物理显存就永远无法归还给 `CUDACachingAllocator`**！攒够 32 个微步，原本只需 **`4 MB`**（`32 × 128 KB`）的数据，硬生生钉死了 **`32.00 GB`** 显存（在 `16K` 序列长度下更高达 **`64.00 GB`，放大 `16,384 倍`**）！
  2. **底层真相二：在 `lm_head` 之后再切片 `lm_head(H)[:, -1, :]`，白白多算 `8,191` 倍矩阵乘并多占 `9.96 GB` 峰值显存（Tier 1）**！
     - 在 Prefill 阶段（以及不需要 `prompt_logprobs` 的常规生成中），我们只需要 **最后一个 Token** 的 Logits 来采样第 1 个生成词。
     - 如果写成 `lm_head(hidden_states)[:, -1, :]`，GPU 会先把全部 `16 × 8,192 = 131,072` 个 Token 乘上 `152,064` 列的词表矩阵（耗时 **`40.80 ms`**，并在显存里瞬间分配出一块 **`9.96 GB`** 的全序列 Logits 巨型张量），然后再切出最后 1 列扔掉前面 8,191 列！
     - 反过来，如果 **先把最后一个 Token 的隐状态切出来再送进 `lm_head`**：`lm_head(hidden_states[:, -1:, :])`（即 HuggingFace 新引入的 `logits_to_keep=1`），`lm_head` 只需要算 `16 × 1 = 16` 个 Token，耗时从 `40.80 ms` 暴降至 **`0.08 ms`（提速 `510.0 倍`）**，峰值 Logits 显存从 `9.96 GB` 暴降至 **`1.2 MB`**！
  采用 **先收集末尾 Token 再进 `lm_head`（`logits_to_keep=1` 或 `hidden_states[cu_seqlens[1:] - 1]`）+ 对所有跨作用域保存的切片强制调用 `.clone()` 剥离底层 `UntypedStorage` + 在 `saved_tensors_hooks` 中开启存储膨胀比审计（Tier 4）**，可将 32 步驻留显存从 `32.00 GB` 压降至 **`0.004 GB`（`4 MB`，缩减 `8,000 倍`）**，Prefill 输出头提速 **`510.0 倍`**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：如何用 1 行代码照出任何张量背后的“隐形物理显存吸血鬼”？**
   - 永远不要只信 `tensor.numel() * tensor.element_size()`（即 `tensor.nbytes`）！那只是逻辑视图大小。
   - 真正占用 GPU 物理显存的是 **`tensor.untyped_storage().nbytes()`**！
   - 定义 **存储钉死放大比（Storage Pinning Amplification Ratio）**：
     $$\rho_{\text{pin}}(t) = \frac{t.\text{untyped\_storage}().\text{nbytes}()}{\max(t.\text{nbytes}, \; 1)}$$
     - 对于健康的独立张量，$\rho_{\text{pin}}(t) = 1.0$；
     - 对于 `last_h = H[:, -1, :]`（当 $S = 8,192$ 时），`last_h.nbytes == 128 KB`，而 `last_h.untyped_storage().nbytes() == 1.00 GB`，**$\rho_{\text{pin}} = 8,192\times$**！
   - **架构师铁律**：任何即将被放入列表、缓存、返回值、跨进程队列或 `ctx.save_for_backward()` 的张量 $t$，只要满足 $\rho_{\text{pin}}(t) > 2.0$，**必须立即调用 `t = t.clone()`**！`.clone()` 会分配一块恰好等于 `128 KB` 的全新独立 `UntypedStorage`，从而让原来那块 `1.00 GB` 的母体显存立即因引用计数归零而被释放！
2. **为什么在一维紧凑布局（Ragged `cu_seqlens`，如 `vLLM / SGLang`）下用 `index_select` 取最后一个 Token 天然不会泄漏？**
   - 在一维紧凑布局 `hidden_states[total_tokens, D]` 中，每个请求最后一个 Token 的下标数组是 `last_indices = cu_seqlens[1:] - 1`（一维整型 Tensor）。
   - 在 PyTorch 中，基础切片 `H[:, -1, :]` 返回的是 **共享存储的视图（View）**，而高级花式索引 `H[last_indices]`（即 `torch.index_select(H, 0, last_indices)`）**天然会分配一块全新的紧凑内存并拷贝那 $B$ 行数据**！因此既完成了 `lm_head` 前置降维（`510x` 加速），又自动剥离了母体存储！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：在自定义 `torch.autograd.Function`（如手写融合 RoPE、MLA 或 Gated Linear Attention）中，什么情况下 `ctx.save_for_backward()` 会踩中这个 `UntypedStorage` 钉死陷阱？
- 典型事故现场：前向算子先做了一次大线性投影得到打包张量 `qkv = W_qkv(x)`（形状 `[B, S, 3 * H * D]`，占 `768 MB`），然后用切片拆出 `q, k, v = qkv.chunk(3, dim=-1)`，甚至只切出其中 64 维的位置编码部分 `k_pe = k[..., :64]`（逻辑大小只有 `16 MB`），最后写了 **`ctx.save_for_backward(k_pe)`**！
- 你以为每层只保存了 `16 MB` 的 `k_pe`，实际上因为 `k_pe` 是 `qkv` 的切片视图，**它把整个 `768 MB` 的 `qkv` 母体张量在前向 80 层里全部钉死在显存里无法释放**（80 层平白无故多吃 **`60 GB` 显存**）！
- **解法**：对保存在 `ctx` 里的窄切片调用 `ctx.save_for_backward(k_pe.clone())`，或者挂载全局调试钩子自动检测：
  ```python
  def pack_hook(t: torch.Tensor) -> torch.Tensor:
      if t.untyped_storage().nbytes() > t.nbytes * 2:
          return t.clone()
      return t
  ```

### Q2：如果把一个大张量的切片 `t_slice = big_tensor[:10]` 通过 `torch.multiprocessing.Queue` 发送给另一个进程，或者调用 `torch.save(t_slice, "ckpt.pt")`，会发生什么？
- **又是一个惊天大坑**！PyTorch 的 `torch.save()` 和跨进程共享在序列化一个视图张量时，默认会 **把整个底层 `UntypedStorage`（例如整块 `4 GB` 内存）全部写进磁盘文件或通过 IPC 映射过去**！所以你在 `torch.save` 任何切片出的子权重或特征之前，也必须调 `.clone()`！

### Q3：`.detach()` 和 `.clone()` 有什么区别？只调 `last_h = hidden_states[:, -1, :].detach()` 能释放那 `1.00 GB` 显存吗？
- **完全不能！**
  - `.detach()` 只是切断了 **Autograd 反向计算图（`grad_fn`）** 的连接，但它返回的张量 **100% 依然与原张量共享同一个 `UntypedStorage` 物理显存块**（`x.detach().untyped_storage().data_ptr() == x.untyped_storage().data_ptr()`）！
  - 想要释放底层大块物理显存，必须调用 **`.clone()`**（如果既要切断计算图又要剥离物理存储，标准连招是 **`x[:, -1, :].detach().clone()`**）！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 PyTorch 张量切片生命周期与 `lm_head` 前置裁剪选型决策树

```mermaid
flowchart TD
    A["大模型前向输出全序列隐状态 hidden_states [B=16, S=8192, D=4096] (占 1.00 GB HBM)"] --> B{"如何计算首字生成 Logits 与缓存末位隐状态?"}
    B -- "先调 lm_head(H) 再切片 [:, -1, :] 且直接缓存切片视图" --> C["⛔ Tier 1: 白算 9.96 GB 全序列 Logits (耗时 40.80 ms) + 32 步切片钉死 32.00 GB 物理显存!"]
    B -- "先切片 H[:, -1:, :] 再进 lm_head, 但未调 .clone() 直接缓存视图" --> D["⛔ Tier 2: lm_head 虽快 (0.08 ms), 但每个 128 KB 切片仍钉死 1.00 GB UntypedStorage -> 狂漏 32 GB!"]
    B -- "对输出切片调 .clone(), 但自定义算子 save_for_backward 仍存未克隆窄视图" --> E["⚠️ Tier 3: 前向激活值被 qkv 窄切片钉死, 额外浪费 8.40 GB HBM!"]
    B -- "lm_head 前置收集 + 跨作用域切片强制 .detach().clone() + 存储比审计钩子" --> F["🏆 Tier 4: lm_head 仅 0.08 ms (快 510 倍) + 32 步缓存仅占 0.004 GB (省 8,000 倍显存)!"]
```

### 3.2 `128 KB` 切片视图钉死 `1.00 GB` `UntypedStorage` vs. `.clone()` 物理存储解绑对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 128 KB 视图壳子死死拽住 1.00 GB 底层 UntypedStorage 不放!"]
      B1["hidden_states [16, 8192, 4096] -> 分配 1.00 GB UntypedStorage (ref_count = 1)"] --> B2["last_h = H[:, -1, :] (128 KB 视图) -> 指向同一块 1.00 GB Storage (ref_count = 2)"]
      B2 --> B3["del H 后 ref_count 仍为 1 -> 1.00 GB 显存永久泄漏! 32 步狂吞 32.00 GB!"]
    end
    subgraph Good["🏆 Tier 4: .detach().clone() 分配 128 KB 独立存储 + Pre-lm_head 投影"]
      G1["last_h = H[:, -1, :].detach().clone() -> 新建 128 KB 紧凑 UntypedStorage"] --> G2["del H 后原 1.00 GB Storage 的 ref_count 归零 -> 立即 100% 释放回显存池!"]
      G2 --> G3["lm_head(last_h) 仅计算最后 1 个 Token -> 0.08 ms (快 510 倍), 32 步仅占 4 MB!"]
    end
```

### 3.3 `UntypedStorage` 引用计数解绑与 `saved_tensors_hooks` 自动防御时序图

```mermaid
sequenceDiagram
    participant Fwd as Transformer Forward Pass
    participant Alloc as PyTorch CUDACachingAllocator (UntypedStorage)
    participant Hook as saved_tensors_hooks / Output Cloner
    participant LM as lm_head Projection

    Fwd->>Alloc: 1. Allocate `H [16, 8192, 4096]` -> `Storage_1GB` (`ref_count = 1`)
    Fwd->>Hook: 2. Slice `view_h = H[:, -1, :]` (`logical = 128 KB`, `Storage_1GB.ref_count = 2`)
    Hook->>Alloc: 3. Detect `storage.nbytes (1GB) > 2 * view.nbytes (128KB)` -> execute `compact_h = view_h.clone()`!
    Alloc->>Alloc: 4. Allocate `Storage_128KB` for `compact_h`; drop `view_h` & `H` -> `Storage_1GB.ref_count == 0` (Freed!)
    Hook->>LM: 5. Run `lm_head(compact_h)` on `[16, 1, 4096]` in 0.08 ms (510x faster than full sequence!)
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 张量切片与 `lm_head` 投影方案 | 32 步缓存驻留显存 | Prefill `lm_head` 耗时 | 物理/逻辑存储放大比 | 适用评价 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: `lm_head(H)[:, -1, :]` + 缓存视图** | 全序列投影后再切片，不调 `.clone()` | `32.00 GB` (严重泄漏! ✕) | `40.80 ms` (`9.96 GB` ✕) | `8,192x` ✕ | ⛔ **算力与显存双灾难** |
| **Tier 2: `H[:, -1, :]` 前置切片 (未调 `.clone()`)** | 加速了 `lm_head`，但视图仍钉死 `1GB` 母体 | `32.00 GB` (严重泄漏! ✕) | `0.08 ms` 🏆 | `8,192x` ✕ | ⛔ **最隐蔽的 OOM 元凶** |
| **Tier 3: 仅克隆输出 (漏掉 `save_for_backward` 窄视图)** | 输出无泄漏，但反向图钉死打包 `QKV` | `8.40 GB` ✕ | `0.08 ms` 🏆 | `64x` ✕ | ⚠️ **长序列训练仍浪费显存** |
| **Tier 4: 前置收集 + `.clone()` 解绑 + 存储比门控 (推荐)** | **先取末位 + `.clone()` 独立存储 + 钩子审计** | **`0.004 GB` (`4 MB` 🏆)** | **`0.08 ms` (`510.0x` 🏆)** | **`1x` (零膨胀 🏆)** | 🏆 **PyTorch 显存管理黄金准则** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/pytorch_tensor_view_slice_untyped_storage_pinning_memory_leak_clone/run.py`](../../experiments/pytorch_tensor_view_slice_untyped_storage_pinning_memory_leak_clone_selection/run.py) · 封存数据：[`results.json`](../../experiments/pytorch_tensor_view_slice_untyped_storage_pinning_memory_leak_clone_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/pytorch-tensor-view-slice-untyped-storage-pinning-memory-leak-clone-sizer.html`](../reports/pytorch-tensor-view-slice-untyped-storage-pinning-memory-leak-clone-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/pytorch-tensor-view-slice-untyped-storage-pinning-memory-leak-clone-selection-cards.html`](../reports/pytorch-tensor-view-slice-untyped-storage-pinning-memory-leak-clone-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Post-lm_head Slice (Uncloned)", "retained_hbm_gb": 32.00, "lmhead_ms": 40.80 },
      { "tier": "Tier 2: Pre-lm_head Slice (Uncloned)", "retained_hbm_gb": 32.00, "lmhead_ms": 0.08 },
      { "tier": "Tier 3: Uncloned Autograd View", "retained_hbm_gb": 8.40, "lmhead_ms": 0.08 },
      { "tier": "Tier 4: Pre-lm_head Gather + .clone()", "retained_hbm_gb": 0.004, "lmhead_ms": 0.08 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "retained_hbm_gb": "Quantity",
    "lmhead_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Pre-lm_head Gather + .clone() Detachment Eliminates 32.0 GB of Pinned View HBM Leakage",
    "subtitle": "Retained HBM across 32 cached steps (GB) across 4 tensor slicing tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "retained_hbm_gb",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有大模型 Prefill 首字生成（`logits_to_keep=1`）、Embedding / Reranker 池化层、EAGLE 投机解码特征缓存、RL 经验回放池以及自定义 `torch.autograd.Function`。
- **⛔ Not for**：把切片视图 `x[:, -1, :]` 直接塞进长期存活的列表/字典/缓存，或先算完整序列 `lm_head(H)` 再切片最后 1 个 Token。

### 生产核验 Checklist
- [ ] **在无需 `prompt_logprobs` 的 Prefill 步中强制开启 `logits_to_keep=1`（先对 `hidden_states` 取最后一个 Token 再调用 `lm_head`）**。
- [ ] **对任何跨函数/跨步缓存或调用 `torch.save` 的张量切片强制执行 `.detach().clone()`**（绝不能只调 `.detach()`）。
- [ ] **在测试与调试环境中用 `saved_tensors_hooks` 校验 `t.untyped_storage().nbytes() <= 2 * t.nbytes`**：自动揪出所有钉死巨型母体显存的窄视图。
