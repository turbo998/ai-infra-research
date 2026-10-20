# FlashInfer GDN Prefix-Cache 状态物化、Cake W8A8 MegaMoE EP16 FC2 Evict-First 与 Cake FMHA 均匀门控 L2 流提示深度解析

> **English Version**: [flashinfer-gdn-prefix-state-w8a8-megamoe-evict-fmha-l2-stream.en.md](flashinfer-gdn-prefix-state-w8a8-megamoe-evict-fmha-l2-stream.en.md) | **返回索引**: [README.md](../../README.md) | **7-Card 评测报告**: [ai-infra-phase33-benchmark-cards.html](../reports/ai-infra-phase33-benchmark-cards.html)

## 1. 一手源码变更矩阵（2026-10-20）

| 组件 | 仓库与 Commit SHA | 发布日期 | 核心文件与函数 | 证据级别 |
| :--- | :--- | :--- | :--- | :--- |
| **FlashInfer GDN Prefix-Cache State Materialization** | `flashinfer-ai/flashinfer` [`99406bb9cae63a7154dc3498460f4778f1b8a33b`](https://github.com/flashinfer-ai/flashinfer/commit/99406bb9cae63a7154dc3498460f4778f1b8a33b) (`#5073`) | `2026-10-20` | `flashinfer/gdn_kernels/gdn_prefix_materialize.py::gdn_prefix_materialize`, `tests/gdn/reference_gdn_materialize.py` | `implementation` + `own-offline` |
| **FlashInfer Cake W8A8 MegaMoE EP16 FC2 Evict-First L2 Hint** | `flashinfer-ai/flashinfer` [`60213537ce15b522fa75e4526187f505ec36ff70`](https://github.com/flashinfer-ai/flashinfer/commit/60213537ce15b522fa75e4526187f505ec36ff70) (`#6857`) | `2026-10-20` | `flashinfer/experimental/cake_w8a8_megamoe_ep16/csrc/cake_w8a8_megamoe_ep16/cake_w8a8_megamoe_ep16_fc2_kernel.cu::kernel_cake_w8a8_megamoe_ep16_fc2` | `implementation` + `author-benchmark` + `own-offline` |
| **FlashInfer Cake FMHA Balanced Decode Uniform-Gated L2 Stream Hints** | `flashinfer-ai/flashinfer` [`566812deed79eff36b6c1298be9e9ea2d4f101e9`](https://github.com/flashinfer-ai/flashinfer/commit/566812deed79eff36b6c1298be9e9ea2d4f101e9) (`#6111`) | `2026-10-20` | `csrc/cake_fmha/cuda/decode_balanced_16bit/default.cu`, `csrc/cake_fmha/cuda/decode_balanced_hd64/default.cu` | `implementation` + `own-offline` |

---

## 2. GDN Prefix-Cache 状态物化算子（`gdn_prefix_materialize`）

### 2.1 为什么 u-cache 异步 Flush 无法直接服务 Prefix Cache？
在 Gated DeltaNet（GDN）混合线性注意力解码中，u-cache 方案不在每个 token 步都就地改写高维 SSM 状态池，而是将上一次 flush 时的检查点状态 $S_0 \in \mathbb{R}^{d_v \times d_k}$ 存放在状态池槽位 `src_slots` 中，同时将后续已接受 token 的 L2 归一化键 $k_j$、Delta-rule 修正项 $u_j$ 以及累积对数衰减 $g_j$ 追加到小容量环形缓冲区（ring）内。

然而，Prefix Cache 要求缓存的状态严格对齐到固定块边界（例如每 $128$ 个 token 一个块），而 u-cache 解码算子的 flush 位置取决于投机解码（MTP）接受长度或调度步长。例如：上次 flush 将检查点停在 token $120$，随后解码在 ring 中追加了 token $121\dots 129$（共 $\text{hist\_len} = 9$ 个条目）。此时 token $128$ 跨过了 Prefix Cache 块边界，需要精确提取 token $128$ 时刻的 SSM 状态写入独立的缓存池槽位 `dst_slots`，且**不能破坏正在继续解码的 `src_slots` 与 ring**。

### 2.2 局部非破坏性 Ring 折叠公式与 Persistent Grid 调度
Commit `99406bb9cae6` 实现了 `gdn_prefix_materialize`，对每个跨边界请求仅截取前 $C = \text{count} \le \text{hist\_len}$ 个 ring 条目（上例中 $C = 128 - 120 = 8$，跳过第 $9$ 个条目 token $129$），执行精确闭式折叠：

$$S_C = \exp(G_C)\, S_0 + \sum_{j=0}^{C-1} \exp(G_C - g_j)\, (u_j \otimes k_j), \qquad G_C = g_{C-1}$$

由于 u-cache ring 中的每个 $u_j$ 已经是 WY 变换求解后的精确单 token Delta-rule 修正向量，截断到任意前缀 $C$ 与逐步递推 $S^{(t)} = \exp(\Delta g_t) S^{(t-1)} + u_t k_t^\top$ 完全等价（本仓库确定性实验验证最大绝对误差为 `0.0`，而若不截断直接做全窗口折叠会产生 `0.130181` 的状态漂移）。

更重要的是，在线推理中 batch 被填充到固定大小（如 $B=256, H_V=64$），但同一步内通常仅有极少数（甚至 $0$ 个）请求恰好跨过 $128$-token 边界：
- 旧版朴素 `1-CTA-per-item` 网格在 `active=0` 时仍需启动全部线程块，纯调度开销高达 **$32.0\,\mu\text{s}$**（`author-benchmark` on B200）。
- `gdn_prefix_materialize` 采用 **Persistent Grid + 设备端 `num_active` 循环上限**（负槽位或负 `count` 行直接跳过），将 $B=256, H_V=64$ 下 `active=0` 的空转开销压降至 **$2.5\,\mu\text{s}$**（降低 **$12.8\times$**）。

---

## 3. Cake W8A8 MegaMoE EP16 FC2 权重 TMA `evict_first` L2 提示

### 3.1 32 Experts/Rank 下 FC2 流式权重的 L2 污染问题
`sm103_mxfp8_mxfp8_bf16_cake`（`cake_w8a8_megamoe_ep16`）面向 $16 \times$ NVIDIA GB300 (`sm_103a`) NVL72 域，固定运行 16 ranks、每 rank 32 个专家、$N = 3072$（hidden）、$K = 5120$（intermediate）、top-8 路由。其第 4 个生成算子 `kernel_cake_w8a8_megamoe_ep16_fc2` 将 SwapAB FC2 GEMM 与路由 scatter 融合。

在单 rank 上，32 个专家的 MXFP8 权重矩阵 `A`（每专家 $3072 \times 5120 \times 1\,\text{B} = 15.0\,\text{MiB}$）与 E8M0 块缩放因子 `SFA`（每专家 $0.46875\,\text{MiB}$）合计达 **$495.0\,\text{MiB}$**。这些流式权重在单次 MoE 层前向中只被消费一次，若使用默认 TMA 4D 加载策略，会将 L2 中可复用的路由激活与跨 rank combine 同步状态全部冲刷出缓存。

### 3.2 `0x12F0000000000000ULL` PTX `L2::cache_hint` 注入
Commit `60213537ce15` 将 `cake_w8a8_megamoe_ep16_fc2_kernel.cu` 中所有专家权重 `A` 与缩放因子 `SFA` 的 `tma_4d_gmem2smem` 调用替换为显式内联 PTX：

```cuda
asm volatile(
    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
    :: "r"(smem_a_addr + (unsigned int)(pf_k * 21504)),
       "l"((&A)), "r"(0), "r"(0), "r"(pf_k),
       "r"(pf_batch_idx * (N / 128) + pf_off_m / 128),
       "r"(tma_full_addr + (pf_k) * 8),
       "l"(0x12F0000000000000ULL)
    : "memory");
```

描述符 `0x12F0000000000000ULL` 将每次 $21{,}504\,\text{B}$ 的 4D TMA 传输标记为 `evict_first`：
- 在 $16 \times$ GB300 (`sm_103a`, CUDA 13.3) CUDA Graph 重放下，完整 MoE 层前向耗时在 $128\dots 4096$ tokens/rank 六组形状上实现 **几何平均 $0.991\times$（降低约 $0.8\%\text{–}1.4\%$，单形状最低 $0.986\times$）**（`author-benchmark`）。
- 本仓库确定性 L2 驻留模型表明，将 $495.0\,\text{MiB}$ 流式 FC2 权重设为 `evict_first` 后，路由激活工作集（$18.5\,\text{MiB}$）的等效 L2 命中率从 `15.0%` 提升至 `94.0%`（`own-offline`）。

---

## 4. Cake FMHA 平衡解码算子的均匀门控 L2 流提示

Commit `566812deed79`（`#6111`）针对 SM100a/SM103a 上的设备端负载均衡（`cake_fmha` balanced）分页 KV 解码算子（`decode_balanced_16bit/default.cu` 与 `decode_balanced_hd64/default.cu`），在每个工作令牌的第 12 个字（`unsigned int plan_flags = work_token_words[base + 11]`）中引入了**均匀性门控（Uniform-Gated）L2 缓存提示**：
1. **均匀切分块 `evict_first`**：当 batch 内所有请求序列长度相同且所有 tile 切分方式一致时，切分 tile（split tiles）通过运行时缓存策略寄存器以 `evict_first` 加载 K/V 页，防止跨 CTA 切分块污染 L2。
2. **完整长序列 V 页预取（仅 $d=64$）**：对于不切分的完整长序列工作项，在 `head_dim == 64` 算子中提前将 V 页预取至 L2。
3. **重度均匀计划与混合批次自动关断**：当 $d=128$ 且单 CTA 承载 $\ge 64$ 个 256-token 配对（heavy uniform plan），或批次内序列长度异构（mixed lengths）时，`plan_flags` 自动清零回退默认策略，避免预取流量反噬 HBM 带宽。

---

## 5. 可复现实验与架构师行动指南

```bash
python experiments/flashinfer_gdn_materialize_megamoe_fc2_fmha_l2/run.py --check
```

- **架构师选型建议**：
  1. 在混合 GDN/Mamba + Attention 推理引擎（如 vLLM / SGLang）启用 Prefix Caching 时，切勿在每个块边界强制同步 flush u-cache；应调用非破坏性 `gdn_prefix_materialize` 算子，以 $2.5\,\mu\text{s}$ 级常驻网格开销在后台物化对齐状态。
  2. 在 Blackwell (`sm_100a` / `sm_103a`) 编写大专家数 EP GEMM 或分页 KV 解码算子时，必须区分“跨算子复用的激活/归约缓冲”与“单次流式扫过的专家权重/长 KV 分片”，对后者显式绑定 `0x12F0000000000000ULL` (`evict_first`) TMA 缓存策略描述符。
