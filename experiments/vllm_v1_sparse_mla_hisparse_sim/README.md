# vLLM V1 FlashMLA Sparse 多代 KV 格式、`HiSparse` 热缓存与 `GDN` 检查点算例 / vLLM V1 Sparse MLA, HiSparse & GDN Simulation (`vllm_v1_sparse_mla_hisparse_sim`)

## 中文概述 (Chinese Overview)

本实验基于 `vllm-project/vllm` (`commit 5285b9b8646d6d0e8b869c9f5c6d277ccfce637b`) 源码中的 `vllm/v1/attention/backends/mla/flashmla_sparse.py`、`vllm/v1/attention/backends/mla/index_group.py`、`vllm/v1/hisparse/runtime.py` 与 `vllm/v1/attention/backends/gdn_attn.py`，复现并验证：
1. **四代 Sparse MLA KV Cache 字节布局**：
   - DeepSeek V3.2 `fp8_ds_mla`：**`656 B/token`**（`512B NoPE + 16B FP32 scales + 128B BF16 RoPE`，压缩比 **`1.756x`**）
   - DeepSeek V4 分页分离布局：**`584 B/token`**（`448B NoPE + 128B BF16 RoPE + 8B UE8M0 scales`，TMA 数据步长 `576B`，压缩比 **`1.973x`**）
   - DeepSeek V4.1 MXFP8：**`528 B/token`**（`512B FP8 [含 RoPE 量化] + 16B UE8M0 scales`，压缩比 **`2.182x`**）
   - Blackwell SM100 `nvfp4_ds_mla`：**`352 B/token`**（`256B E2M1 NoPE + 64B 无缩放 FP8 RoPE + 32B 置换 FP8 scales`，通过 `8 * (s & 3) + (s >> 2)` 映射使单线程 8 个缩放因子落入连续 `0..7` 字节，压缩比 **`3.273x`**，32×128K 集群 KV 显存从 `274.50 GiB` 压至 `83.88 GiB`）。
2. **NVIDIA `HiSparse` 热缓存与驻留解析共享内存护栏**：默认热缓存 `device_buffer_size = (max_query_len + 1) * model_top_k`（`q=1` 为 `4,096` 行、`q=4` MTP 为 `10,240` 行，将 128K 上下文 GPU HBM 占用再压缩 **`12.8x`** 至 `7.8125%`），并精确计算驻留解析器共享内存 `required = 4 * (2 * (max_union_rows + 1) + 3 * ceil(hot_size / 32) + 7) + 2 * hot_size`（`q=4` 需 `89,892 B`）及 `SparseMLAIndexGroup` 旁路流单次转换（减少 **`61x`** 转换 Kernel 发射）。
3. **`GDNAttentionBackend` FlashInfer 64-Token 对齐检查点规划**：验证 `every_n_tokens = gcd(*offsets) = 64`、`cu_starts = [0, 4, 12, 16]` 与状态行映射 `rows = [0, 1, 7, 14]`。

## English Overview

This deterministic Python 3.12 standard-library experiment models `vllm-project/vllm` (`commit 5285b9b8646d6d0e8b869c9f5c6d277ccfce637b`):
1. **Four-Generation FlashMLA Sparse KV Formats (`flashmla_sparse.py`)**: Compares BF16 (`1,152 B/tok`) against DeepSeek V3.2 (`656 B/tok`), DeepSeek V4 (`584 B/tok`, `576B` split-page TMA stride), DeepSeek V4.1 MXFP8 (`528 B/tok`), and SM100 `nvfp4_ds_mla` (`352 B/tok`, **`3.273x`** compression, verifying that `8 * (s & 3) + (s >> 2)` maps a thread's 8 logical scale blocks `[0, 4, 8, 12, 16, 20, 24, 28]` to contiguous bytes `[0..7]`).
2. **`HiSparse` Hot-Cache Sizing & Shared-Memory Residency Guardrail (`hisparse/runtime.py` & `index_group.py`)**: Sizes the GPU hot buffer at `(max_query_len + 1) * model_top_k` (`4,096` rows for `q=1`, `10,240` rows for `q=4` MTP, cutting 128K GPU HBM residency by **`12.8x`** to **`7.8125%`**), verifies the exact shared-memory budget (`89,892 B` at `q=4`), and reduces logical-to-physical top-k index conversion launches by **`61x`** via `SparseMLAIndexGroup._convert_once`.
3. **`GDN_ATTN` FlashInfer 64-Token Checkpoint Planner (`gdn_attn.py`)**: Verifies `every_n_tokens = gcd(*offsets) = 64`, `cu_starts = [0, 4, 12, 16]`, and `rows = [0, 1, 7, 14]`.

## 运行与校验 / Run & Verify

```powershell
python experiments\vllm_v1_sparse_mla_hisparse_sim\run.py --check
```
