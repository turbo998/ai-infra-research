# vLLM V1 FlashMLA Sparse 四代 KV 布局、NVIDIA `HiSparse` 分层热缓存与 `GDN` 混合检查点架构

> **Language / 语言**: **中文 (Chinese)** | [English (英文版)](vllm-v1-sparse-mla-hisparse-gdn-hybrid.en.md)

## 1. 架构背景与核心痛点：超长上下文稀疏 MLA 与混合线性注意力的显存/调度工程

在 128K–1M 超长上下文下，DeepSeek-V3.2 / V4 / V4.1 的稀疏多头潜在注意力（Sparse MLA）与 GatedDeltaNet（`GDN_ATTN`）混合状态空间模型面临三大系统级挑战：
1. **跨代 Sparse MLA KV Cache 的字节对齐、RoPE 量化与反量化线程访存合并**：从 V3.2 (`656 B`)、V4 (`584 B`)、V4.1 (`528 B`) 到 Blackwell SM100 `nvfp4_ds_mla` (`352 B`)，如何在极限压缩 KV 字节数的同时保证 TMA 2D/3D 步长对齐与单线程反量化缩放因子的连续 8 字节合并读取？
2. **主机卸载（Host-Backed Sparse KV）下的 GPU 热缓存（Hot Cache）容量与共享内存边界**：在 NVIDIA `HiSparse` 运行时中，投机解码（如 MTP `max_query_len = 4`）每步需要合并多个验证 Token 的 Top-$K$ 集合，若热缓存过小将触发缺页崩溃，若过大则会超出 SM 共享内存（Shared Memory）驻留解析器的硬上限或 `int16` 槽位索引上限（`32,768`）；
3. **多层共享索引转换与 GatedDeltaNet Prefill 内部状态检查点对齐**：61 层 Sparse MLA 若逐层重复执行逻辑 Top-$K$ 到物理块表索引转换，或分段捕获 CUDA Graph 时误用图内局部 `cudaEventRecord`，将导致严重性能回退乃至 `cudaErrorInvalidValue` 崩溃。

基于 **`vllm-project/vllm` (`commit 5285b9b8646d6d0e8b869c9f5c6d277ccfce637b`)** 最新合入的 `vllm/v1/attention/backends/mla/flashmla_sparse.py`、`vllm/v1/attention/backends/mla/index_group.py`、`vllm/v1/hisparse/runtime.py` 与 `vllm/v1/attention/backends/gdn_attn.py` 一手源码，本文给出完整字节账本与闭式方程。

---

## 2. 一手源码机制拆解 (`implementation`)

### 2.1 `vllm/v1/attention/backends/mla/flashmla_sparse.py`：四代 Sparse MLA KV 字节结构与 `8x4 -> 4x8` 置换

在 `flashmla_sparse.py` (`blob 9722256f889bcedf5ae2ee9a30ee6c7bb7828a06`) 中，vLLM V1 精确定义了四种量化 Sparse MLA KV 格式，并通过每 Token 字节数自动推断底层内核格式：

1. **DeepSeek V3.2 `fp8_ds_mla`（`656 Bytes/token`）**：
   - 前 `512 B`：`512 × float8_e4m3` 量化 NoPE；
   - 中间 `16 B`：`4 × float32` 缩放因子（每 128 个元素 1 个 `fp32` scale）；
   - 末尾 `128 B`：`64 × bfloat16` 未量化 RoPE（保障旋转位置精度）。
2. **DeepSeek V4 分页分离格式（`584 Bytes/token`）**：
   - `448 B` `float8_e4m3` NoPE + `128 B` (`64 × bfloat16`) RoPE + `8 B`（`7 × ue8m0 + 1B pad`，每 64 个元素 1 个 `ue8m0` scale）；
   - **页内数据/缩放分离布局**：单个 Page 先连续存放所有 Token 的 `576 B` 数据行（匹配 Decode 内核 `576 B` TMA 步长），末尾再集中存放所有 Token 的 `8 B` 缩放行。
3. **DeepSeek V4.1 MXFP8 交错格式（`528 Bytes/token`）**：
   - 前 `512 B`：全部 512 维（**含 64 维 RoPE 一并量化为 `float8_e4m3`**）；
   - 末尾 `16 B`：`16 × ue8m0` 缩放因子（每 32 个连续 `float8_e4m3` 1 个 `ue8m0` scale，即标准 `MXFP8`），每 Token 数据与 scale 紧邻交错（TMA 步长 `528 B`）。
4. **Blackwell SM100 `nvfp4_ds_mla`（`352 Bytes/token`）**：
   - 前 `256 B`：`512 × e2m1` NoPE（每字节打包 2 个元素，低半字节为偶数索引）；
   - 中间 `64 B`：`64 × float8_e4m3` **无缩放因子（Unscaled）RoPE**（利用 `e4m3` 4 位指数直接覆盖 RoPE 幅值范围）；
   - 末尾 `32 B`：`32 × float8_e4m3` NoPE 缩放因子（每 16 个元素 1 个 scale），按 **`8×4 -> 4×8` 转置置换存储**：逻辑块 `s`（`0 <= s < 32`）的物理字节偏移为：
     $$\text{byte\_offset}(s) = 8 \times (s \mathbin{\&} 3) + (s \gg 2)$$
     使得单个 FlashMLA 反量化线程所需的 8 个缩放因子（`s = 0, 4, 8, 12, 16, 20, 24, 28`）恰好落在连续物理字节 **`[0, 1, 2, 3, 4, 5, 6, 7]`** 上，单次 64-bit 向量加载即可读完！

### 2.2 `vllm/v1/hisparse/runtime.py` & `index_group.py`：`HiSparse` 热缓存与共享内存驻留解析方程

在 `ResolvedHiSparseConfig.from_vllm_config` (`blob 87916ba4ad945f09b24543d02b33fdcf709f3efd`) 中：
- 单请求单步最多选中的主机行数（并集上限）为：
  $$\text{max\_union\_rows} = Q_{\max} \times K_{\text{top}}$$
- 默认 GPU 热缓存行数留出 1 个 $K_{\text{top}}$ 的 LRU 余量，且受限于 `int16` 槽位索引上限 `32,768`：
  $$H_{\text{hot}} = (Q_{\max} + 1) \times K_{\text{top}} \le 32768$$
- `_check_residency_shared_memory` 严格校验 `hisparse_resolve_residency` 内核所需的每 CTA 共享内存（包含并集哈希键值表、分块偏移、5 个计数器、完成位图与 `int16` 压缩 LRU）：
  $$\text{required\_smem\_bytes} = 4 \times \left(2(\text{max\_union\_rows} + 1) + 3\left\lceil \frac{H_{\text{hot}}}{32} \right\rceil + 7\right) + 2 H_{\text{hot}}$$

同时，在 `SparseMLAIndexGroup._convert_once` (`blob ba787b38da552acbd58c58ca9b2f6a5f00b85fc6`) 中，所有共享同一 Indexer 的 MLA 层只在 `layer_index == 0` 时于 `side_stream` 上执行一次 `triton_convert_req_index_to_global_index`；且当处于分段 CUDA Graph（`in_piecewise_cudagraph()`）时，自动跳过图内局部 `logical_topk_ready.record()` 并改用 `wait_stream`，规避跨图/Eager 边界等待局部 Event 引发的 `cudaErrorInvalidValue`。

### 2.3 `vllm/v1/attention/backends/gdn_attn.py`：FlashInfer GDN 64-Token 检查点规划

在 `plan_flashinfer_checkpoints` (`blob 660ef5387b1752953adf8dbea8f8f81cfb8f680f`) 中，由于 FlashInfer GDN Prefill 内核按 64-Token 分块整数倍发射中间状态（`FLASHINFER_GDN_CHECKPOINT_ALIGNMENT = 64`），规划器通过最大公约数确定统一发射步长：

$$\text{every\_n\_tokens} = \gcd(\text{offsets}), \quad \text{cu\_starts}[i+1] = \text{cu\_starts}[i] + \left\lfloor \frac{\text{query\_len}_i}{\text{every\_n\_tokens}} \right\rfloor$$

$$\text{rows}[\text{first} + i] = \text{cu\_starts}[i] + \frac{\text{offset}_i}{\text{every\_n\_tokens}} - 1$$

---

## 3. 原创确定性离线实验 (`own-offline`) 与量化结论

通过 [`experiments/vllm_v1_sparse_mla_hisparse_sim/run.py`](../../experiments/vllm_v1_sparse_mla_hisparse_sim/run.py) 对 61 层、`32 × 128K`（`4,194,304` tokens）集群 KV 显存、`HiSparse` 热缓存与 `GDN` 检查点进行确定性验证（见 [`results.json`](../../experiments/vllm_v1_sparse_mla_hisparse_sim/results.json)）：

| Sparse MLA KV 格式 / 机制 | 每 Token 字节 (`B`) | TMA 步长 (`B`) | `32×128K` (61L) 显存 (`GiB`) | 相对 BF16 压缩比 | 核心硬件/算法特征 |
|---|---|---|---|---|---|
| `bf16_mla_baseline` | `1,152 B` | `1,152 B` | `274.50 GiB` | `1.000x` | 512 BF16 NoPE + 64 BF16 RoPE |
| `deepseek_v3_2_fp8_ds_mla` | **`656 B`** | `656 B` | `156.31 GiB` | **`1.756x`** | 512B FP8 NoPE + 16B FP32 Scale + 128B BF16 RoPE |
| `deepseek_v4_fp8_split_page` | **`584 B`** | **`576 B`** | `139.16 GiB` | **`1.973x`** | 页内先排 `576B` 数据行，尾置 `8B` (`7×ue8m0+1B`) 缩放行 |
| `deepseek_v4_1_mxfp8` | **`528 B`** | `528 B` | `125.81 GiB` | **`2.182x`** | RoPE 一并 FP8 量化 + `16×ue8m0` (`gran=32`) 行内交错 |
| `sm100_nvfp4_ds_mla` | **`352 B`** | `352 B` | **`83.88 GiB`** | **`3.273x`** | 256B E2M1 + 64B 无缩放 FP8 RoPE + `8*(s&3)+(s>>2)` 连续 8B Scale |

**`HiSparse` 热缓存与 `SparseMLAIndexGroup` 收益（`K_top = 2048, L_seq = 128K`）**：
- 单步 Decode (`Q_max = 1`)：`H_hot = 4,096` 行，需 **`26,148 B`** SMEM；
- 4-Token MTP 投机解码 (`Q_max = 4`)：`H_hot = 10,240` 行（仅占 128K 上下文的 **`7.8125%`**，GPU HBM 驻留压缩 **`12.8x`**），需 **`89,892 B`** SMEM（完全适配 Hopper 228 KiB / Blackwell 227 KiB SMEM 上限）；
- `SparseMLAIndexGroup._convert_once` 将 61 层索引转换发射从 `61` 次（`0.5795 ms`）压至 **`1` 次**（**`0.0095 ms`**，减少 **`61x`** 发射）。

---

## 4. 解决方案架构师决策指南与可证伪问题

1. **格式选型护栏**：在 Hopper SM90 上优先采用 V4.1 `528 B` MXFP8 或 V3.2 `656 B` `fp8_ds_mla`；在 Blackwell SM100 上启用 `nvfp4_ds_mla` (`352 B/token`, 64-token 固定页块)，可将单机 8×B200 的百万 Token 并发容量提升 **`3.273x`**；
2. **`HiSparse` MTP 投机步数上限**：当 `K_top = 2048` 时，由于 `int16` 槽位索引上限 `32,768`，`(Q_max + 1) * 2048 <= 32768` 要求 **`Q_max <= 15`**，且在 `Q_max = 8` 时 `required_smem_bytes = 174,884 B` 接近 SMEM 上限；
3. **后续可证伪问题**：当混合请求批次中同时包含 `offset = 64` 与 `offset = 192` 的 GDN 前缀检查点时，统一 `gcd = 64` 发射相比按序列独立步长发射会增加多少状态池写带宽？
