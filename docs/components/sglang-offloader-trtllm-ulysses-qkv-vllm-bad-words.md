# SGLang CPU Offloader Storage-View Preservation, TensorRT-LLM Packed 5D QKV Ulysses Unscatter & vLLM Rust Bad-Words Parity

> **Language**: [中文](sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words.md) | [English](sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words.en.md)
> **Phase**: 65 (`priority: 74`) · **Track B**: Inference Serving, Diffusion & Multi-Node Sequence Parallelism
> **Verified Upstream Sources**: `sgl-project/sglang@6d6c9cd4ae` (`#44139`), `sgl-project/sglang@494eeecb6c` (`#45944`), `sgl-project/sglang@f9129915a3` (`#45994`), `NVIDIA/TensorRT-LLM@1586d85375` (`#17147`), `NVIDIA/TensorRT-LLM@8d1c7e0b26` (`#20660`), `vllm-project/vllm@34b2954482` (`#63315`)

---

## 1. 核心架构问题与研究背景

在混合线性注意力（KDA / RadixLinearAttention）、长序列多模态扩散生成（Qwen-Image / Wan TI2V Ulysses Sequence Parallelism）以及 Rust 高并发前端服务中，框架层的 **张量存储别名（Storage Aliasing）**、**TorchDynamo 字典大小守卫（Dict-Size Guards）**、**All-to-All 后处理内存重排** 与 **跨语言 Tokenizer 采样契约一致性** 直接决定了推理系统的正确性与尾延迟：

1. **SGLang CPU Offload 下的缓存权重视图悬空 (`sglang@6d6c9cd4ae`)**：`torch.func.functional_call` 仅替换模块中正式注册的 `nn.Parameter` 与 `Buffer`。然而，许多高性能算子在初始化或权重加载后会把权重的视图（View）或派生张量保存为普通 Python 属性（例如 KDA 的 `attn.conv_weights` 共享 `qkv_conv1d.weight` 的底层 Storage，`GemmaRMSNorm.gemma_weight` 保存 `weight + 1.0`）。当 CPU Offloader 将 `qkv_conv1d.weight` 卸载到 CPU 时，未注册为 Parameter 的 `attn.conv_weights` 视图变成指向已释放/迁出显存的悬空视图（Stale View）。
2. **SGLang Diffusion Linear-GELU 缓存污染 `torch.compile` 追踪 (`sglang@494eeecb6c` & `f9129915a3`)**：`can_use_linear_gelu(linear, x)` 使用全局字典 `_VERDICTS[linear]` 缓存模块级判定结果。当在 `torch.compile` 内逐层追踪时，每遇到一个新的 Linear 层就会向 `_VERDICTS` 插入一项；TorchDynamo 默认对字典长度安装 `DICT_LENGTH` 守卫，导致每编译一层就触发一次全图重编译（$L$ 层触发 $L$ 次重编译）。此外，Wan TI2V 在 Sequence Parallelism (SP) 下未广播随机采样的时间步 `timestep`，导致不同 SP Rank 使用不同去噪时间步 (`f9129915a3`)。
3. **TensorRT-LLM Qwen-Image Ulysses All-to-All 后的 Packed QKV 重排开销 (`TensorRT-LLM@1586d85375`)**：在 Ulysses 序列并行中，自注意力 QKV 投影后经过 All-to-All 得到紧凑的 5D 接收缓冲区 `[P, B, Sp, 3, H, D]`。此前 Python 侧必须先执行 `permute -> contiguous -> unbind` 将其拆分为独立的 `q_in, k_in, v_in`，再调用 `launchUlyssesPostUnscatter`，不仅引入多次额外 Kernel Launch，还使 HBM 读写流量翻倍（4× Tensor 大小）。
4. **vLLM Rust Frontend `bad_words` 与 Python 引擎的语义漂移 (`vllm@34b2954482`)**：vLLM 的 Rust 高性能 HTTP/gRPC 前端在解析 `bad_words` 时，此前未能在分词前拦截空字符串 `""`、未做预分词去重与最大数量上限（`max_num_bad_words`）检查，且前导空白修剪与 `add_prefix_space` 变体保留条件与 Python `sampling_params.py` 存在细微偏差。

---

## 2. 源码级实现机制与公式推导

### 2.1 SGLang 通用底层 Storage 扫描与别名重绑定 (`6d6c9cd4ae` & `494eeecb6c`)

在 `python/sglang/srt/utils/offloader.py` 的 `_get_resident_parameter_ids(module)` 中，SGLang 移除了针对特定层名称的硬编码，改为通用的 **底层 `untyped_storage().data_ptr()` 集合匹配算法**：

1. **收集普通属性张量的 Storage 指纹**：遍历 `module.modules()` 中每个子模块的 `vars(owner).values()`，凡是满足 `isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter) and not v.is_meta and v.numel() > 0` 的属性，将其 `(v.device, v.untyped_storage().data_ptr())` 加入 `cached_storages`。
2. **锁定共享 Storage 的 Parameter 常驻 GPU**：遍历 `module.parameters()`，若 `(param.device, param.untyped_storage().data_ptr()) in cached_storages`，则将该 `id(param)` 加入 `resident` 集合，禁止 Offloader 将其迁出 GPU。
3. **权重重载后的显式刷新钩子 (`loader.py`)**：在 `_rebind_parameter_aliases` 中统一调用 `GemmaRMSNorm.refresh_gemma_weight()` 与 `RadixLinearAttention.refresh_conv_weights(cweight)`。
4. **编译期绕过 `_VERDICTS` (`494eeecb6c`)**：
   ```python
   if torch.compiler.is_compiling():
       # dynamo guards on the cache's size, so reading it recompiles per linear
       return _input_independent_ok(linear)
   ```
   将 $L$ 个 Linear 层的 Dynamo 重编译次数从 $\mathcal{O}(L)$ 严格收敛为 $\mathcal{O}(1)$。

### 2.2 TensorRT-LLM `ulyssesPackedQkvPostUnscatterKernel` 与 Int32 溢出护栏 (`1586d85375`)

在 `cpp/tensorrt_llm/kernels/ulyssesPostUnscatterKernel.cu` 中，新增了直接消费 `[P, B, Sp, 3, H, D]` 缓冲区的 CUDA 算子：
- **线程块与网格映射**：每个线程使用 `uint4`（16 字节 = 8 个 `BF16` 元素，`VEC = 8`）向量化拷贝。设每行向量数 $\text{vec\_per\_row} = D / 8$，则：
  $$\text{blockDim.x} = H \times \frac{D}{8} \le 1024, \quad \text{gridDim} = (3 \cdot P \cdot S_p,\ B,\ 1)$$
- **零中间拷贝的 6D $\to$ 4D 索引直达公式**：令 $\text{PSp} = P \cdot S_p$，$qkv = \lfloor \text{blockIdx.x} / \text{PSp} \rfloor \in \{0, 1, 2\}$，$psp = \text{blockIdx.x} \bmod \text{PSp}$，$p = \lfloor psp / S_p \rfloor$，$sp = psp \bmod S_p$：
  $$\text{in\_base} = \Big(\big(((p \cdot B + b) \cdot S_p + sp) \cdot 3 + qkv\big) \cdot H + h\Big) \cdot D + \text{vec\_idx} \cdot 8$$
  $$\text{out\_base} = \Big((b \cdot \text{PSp} + psp) \cdot H + h\Big) \cdot D + \text{vec\_idx} \cdot 8$$
  根据 $qkv \in \{0, 1, 2\}$ 直接将 `*in_v4` 写入 `q_out`、`k_out` 或 `v_out` 的 `out_base` 偏移处，完全省去 Python 侧 `permute + contiguous + unbind`，使该阶段 HBM 流量精确减半（降低 **50.0%**）。
- **Int32 乘积与求和乘积溢出护栏 (`ulyssesPostUnscatterOp.cpp`)**：新增 `checkInt32Dim`、`checkInt32Product("P*Sp_q", P, Sp_q)` 与 `checkInt32SumProduct("P*(Sp_q+Sp_k+Sp_v)", P, Sp_q, Sp_k, Sp_v)`，在进入 Kernel 启动前拦截超长序列导致的 32 位网格维度截断。

### 2.3 vLLM Rust Frontend `tokenize_bad_words` 严格对齐契约 (`34b2954482`)

在 `rust/src/text/src/lower.rs` 中，Rust 前端实现了与 Python `sampling_params.py` 完全一致的五步确定性管线：
1. **前置空词拦截**：若 `bad_words.iter().any(String::is_empty)`，在调用任何 Tokenizer 之前立即返回 `TokenIdsError::EmptyBadWord`。
2. **输入去重与上限检查**：使用 `HashSet` 在分词前过滤重复词条，并校验词条数不超过 `max_num_bad_words()`。
3. **前导空白剥离与双变体编码**：对 `bad_word.trim_start()` 分别编码无前导空格序列 $T_{\text{no\_sp}}$ 与带前导空格序列 $T_{\text{sp}}$，仅当满足以下充要条件时才追加 $T_{\text{sp}}$：
   $$T_{\text{sp}}[0] \neq T_{\text{no\_sp}}[0] \ \land\ |T_{\text{sp}}| == |T_{\text{no\_sp}}|$$

---

## 3. 确定性实验验证结果 (`experiments/sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words`)

运行 `python experiments/sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words/run.py --check` 通过全部 7 项不变量校验：

| 子系统与验证项 | 优化前 / 错误行为 | 优化后 (`Phase 65` 实测结果) | 关键工程收益 |
|---|---|---|---|
| **SGLang CPU Offloader** | KDA `qkv_conv1d.weight` 被卸载导致 `conv_weights` 视图悬空 | 自动识别共享 `storage_ptr=2002` 并锁定常驻 GPU | 消除 Hybrid/KDA 模型 CPU Offload 崩溃 |
| **SGLang Diffusion `torch.compile`** | `_VERDICTS` 字典长度变化触发 40 次重编译 | `is_compiling()` 分支绕过缓存，仅编译 **1 次** | 编译耗时降低 $40\times$ |
| **TRT-LLM Ulysses Packed QKV** | Python `permute+contiguous+unbind` 产生 `301.99 MB` HBM 流量 | 单 Kernel `uint4` 直写 `q/k/v_out`，流量降至 `150.99 MB` | HBM 流量减少 **50.0%** (`threads=384 <= 1024`) |
| **vLLM Rust `bad_words`** | 缺失空串前置报错、重复词冗余分词与长度不等变体混入 | 空串立即返回 `EmptyBadWord`，精确生成 `[[101], [201], [102, 103], [104, 105]]` | Rust 与 Python 前端 100% 行为等价 |

---

## 4. 面向解决方案架构师的设计建议与可证伪问题

### 架构师检查清单
1. **自定义 Layer 的 CPU Offload 兼容性规则**：在 PyTorch/SGLang 中编写自定义 Attention 或 Norm 层时，若在 `__init__` 或 `process_weights_after_loading` 中缓存了 `Parameter` 的转置/切片视图（非拷贝），必须确保其与原 `Parameter` 共享同一个 `untyped_storage()`，并提供显式 `refresh_*` 方法供权重热更新调用。
2. **Ulysses 序列并行的线程块上限约束**：`ulyssesPackedQkvPostUnscatter` 要求 $H \times (D / 8) \le 1024$。当单卡局部 Head 数 $H > 64$（对应 $D=128$）时，需进一步增大每线程处理的向量数或沿 $H$ 维拆分 `blockIdx.z`。

### 后续可证伪实验问题
- **Q1**: 当 Ulysses 序列并行在 $B=1, P=8, S_p=8192, H=64, D=128$ 时，`threads = 64 * 16 = 1024` 恰好触及线程块上限；若 $D=256$，是否必须将 `blockIdx.z` 扩展为 $H$ 维度以避免触发 `threads <= 1024` 断言？
