# FlashInfer SM12x W4A16 TC-Decode Ultra FC2 強制 Tile 重綁定、vLLM ROCm Kimi-K3 GLA 門控與 MLA FP8 預填充融合及 SGLang Muse Spark 統一 KV 記憶體防護

**English Companion**: [flashinfer-sm12x-w4a16-ultra-fc2-repin-vllm-kimi-k3-rocm-mla-sglang-muse-unified-kv.en.md](flashinfer-sm12x-w4a16-ultra-fc2-repin-vllm-kimi-k3-rocm-mla-sglang-muse-unified-kv.en.md)

## 1. 背景与架构问题

在 Blackwell SM12x（如 DGX Spark GB10 SM121）、AMD CDNA4 (`gfx950`) 与 B200 多模态混合推理栈中，跨算子边界（Custom-Op / Unified Pool / Speculative Draft）的不变量不匹配会导致启动崩溃、冗余带宽消耗或静默 KV 污染：
1. **FlashInfer SM12x W4A16 TC-Decode Ultra FC2 重绑定失败**：对于 `nvidia/Qwen3.6-35B-A3B-NVFP4`（`hidden_size=2048, intermediate_size=512, top_k=8`），小批量解码（`m <= 4`）启用 `tc_decode_fused_sum=True`，选择器将 FC2 Tile 从 `128 -> 256 -> 512` 放宽以单波驻留，此时 256 线程下的 `tile_k = 512 * 32 / 64 // 8 = 32`。然而每次通过 `torch.ops.flashinfer.w4a16_fused_moe_launch` 自定义算子边界时，`_w4a16_fused_moe_launch_flat` 使用 `force_tile_config` 重新调用 `compile_w4a16_fused_moe`，而 `_candidate_tile_fits` 强制执行搜索剪枝下限 `tile_k >= 64`，导致选择器自己选出的 `tile_k=32` 抛出 `ValueError: force_tile_config fc2 tile (tile_k=32, tile_n=512) does not fit problem N/K=2048/512`。
2. **vLLM ROCm Kimi-K3 GLA 门控投影与 MLA FP8 预填充准备开销**：Kimi-K3 启用 `mla_use_output_gate=True` 时，原有的 `g_proj` 作为独立 `ColumnParallelLinear` 重复读取 `[tokens, 7168]` 激活；解码阶段 `q_out` 填充至 AITER 要求的 `32` 头（原生 `24` 头）需要拷贝；CDNA4 预填充阶段的 `k_pe` 拼接、头填充和 FP8 转换散落为 4 个算子启动，且每次调用重复分配标量 `torch.ones(())`。
3. **SGLang Muse Spark 统一 KV 池与 DFLASH FA4 草稿融合冲突**：在 `--enable-unified-memory` 下，KV 池在每个 Token 行内交错存储各层数据，而 `MuseSparkAttention._fused_kv_store` 假设单层连续平面；同时当 `--speculative-draft-kv-cache-dtype` 设为默认 `auto`/`None` 时，`_fused_draft_decision` 跳过了编译期检查，未能识别 DFLASH + FA4 始终使用草稿模型计算精度（`bf16`），与宿主 `fp8`/`fp4` KV 缓存发生布局冲突。

## 2. 一手源码实现路径（Implementation Audit）

### 2.1 `flashinfer-ai/flashinfer` (`fc8cfd004324f5f3dbd54ed4aa50192aa7592888`, `#6292`, `2026-10-20`)
- **`flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py`**：
  - 在 `_candidate_tile_fits` 中新增参数 `enforce_search_floor: bool = True`：自动搜索保持 `tile_n >= 64 and tile_k >= 64 and cta_threads >= 128` 剪枝空间；
  - 在 `compile_w4a16_fused_moe` 的 `force_tile_config` 校验循环中传入 `enforce_search_floor=False`，仅校验**整除性**（`problem_n % tile_n == 0`、`problem_k % tile_k == 0`）、**NVFP4 缩放组对齐**（`tile_k % 16 == 0`）与**共享内存上限**（`smem_bytes <= max_shared_mem`）。
  - **作者实测（`author-benchmark`）**：在 DGX Spark GB10 (SM121) 上运行 `nvidia/Qwen3.6-35B-A3B-NVFP4` GSM8K（1319 题 5-shot），修复后 W4A16 正常服务并取得 **`0.9158` 准确率**（对齐 0.6.16.post3 的 `0.9136` 与 Marlin W4A16 的 `0.9110`），相比被迫降级为 W4A4（`0.8900`）提升 **`+2.58` 个百分点**。

### 2.2 `vllm-project/vllm` (`09fd027217648768c417261ab9a90a3149649ad2`, `#59634`, `2026-10-20`)
- **`vllm/models/kimi_k3/amd/{linear.py,mla.py,mtp.py}`**：
  - 当 `q_lora_rank is not None and config.mla_use_output_gate` 时，使用 `KimiK3MergedQKVGateLinear` 将 `.q_a_proj` (shard 0, `1536`)、`.kv_a_proj_with_mqa` (shard 1, `576`) 与 `.g_proj` (shard 2, `24 * 128 = 3072`) 融合为单次 `5184` 列 GEMM，4096 Token 下每层节省 **`58,720,256 B` (`56.0 MiB`)** 激活读取；
  - 在 `_fused_decode` 中直接分配 `q_out_padded = torch.empty((num_tokens, self._fused_q_heads, head_size), dtype=fp8_dtype)` 并取切片 `q_out = q_out_padded[:, :num_heads]`；在 `rocm_aiter_mla.py` 的 `get_mla_padded_q` 中新增 `if q.stride(0) == m * q.stride(1): return q.as_strided((q.shape[0], m, q.shape[2]), q.stride())` 零拷贝恢复 32 头视图；
  - 在 CDNA4 (`gfx950`) 且 `head_dims` 均为 2 的幂（`not any(d & (d - 1) for d in head_dims)`）时启用 `fused_mla_prefill_qkv_fp8`，单核完成 `k_pe` 拼接、头填充与 FP8 量化，并缓存 `self._fp8_prefill_one_scale`。

### 2.3 `sgl-project/sglang` (`cfec1b91bb33f3267d1811680b51fd581e0df458`, `#45356`, `2026-10-20`)
- **`python/sglang/srt/models/muse_spark.py` & `mem_cache/kv_cache_configurator.py`**：
  - `_fused_kv_store` 首行检查 `if get_memory().enable_unified_memory: return None`，回退到支持跨层交错步长的注意力后端缓存写入路径；
  - `_fused_draft_decision` 将条件扩展为 `if draft_kv_cache_dtype not in (None, "auto") or (self.spec_algorithm.is_dflash_family() and draft_backend == "fa4"):`，提前拦截草稿 `bf16` 与宿主 `fp8`/`fp4` 的非法融合。

## 3. 确定性实验复算（`own-offline`）

运行 `python experiments\flashinfer_sm12x_w4a16_vllm_kimi_rocm_sglang_unified_kv\run.py --check`：

| 子系统 | 关键核验指标 | 修复前 / 基线 | 修复后 (`Phase 37`) | 证据分级 |
|---|---|---|---|---|
| FlashInfer SM12x W4A16 | Ultra FC2 (`32, 512, 256`) 重绑定 | `False` (`ValueError` 崩溃) | **`True`（且拦截 `tile_k=8 / tile_n=768 / smem=1024`）** | `implementation` + `own-offline` |
| Qwen3.6-35B-A3B GSM8K | GB10 (SM121) 精度对比 W4A4 | `0.8900` (W4A4) | **`0.9158` (`+2.58 pts`)** | `author-benchmark` |
| vLLM ROCm Kimi-K3 MLA | QKV+Gate 合并列数 / 节省激活读 | 2 次 GEMM | **1 次 GEMM (`5,184` 列，省 `58.72 MB`/层)** | `implementation` + `own-offline` |
| vLLM CDNA4 MLA Prefill | `q.as_strided` 零拷贝 + 融合准备核 | 4 次算子启动 + 拷贝 | **1 次 `fused_mla_prefill_qkv_fp8` + `0 B` 填充拷贝** | `implementation` + `own-offline` |
| SGLang Muse Spark | 统一池 `_fused_kv_store` / DFLASH FA4 | 启用融合（越界/精度错配） | **`None`（交错池安全回退） / 拒绝异构融合** | `implementation` + `own-offline` |

## 4. 架构师设计练习与可证伪问题

1. **计算练习**：当 `num_tokens = 8192`、`hidden_size = 7168`、61 层 MLA 全部启用 `KimiK3MergedQKVGateLinear` 时，单次前向传播总共减少多少 GiB 的 HBM 激活读取？
   - 答：每层减少 $8192 \times 7168 \times 2 = 117,440,512\text{ B}$，61 层共减少 $7,163,871,232\text{ B} \approx 6.672\text{ GiB}$。
2. **可证伪问题**：若 Kimi-K3 某变体的 `qk_rope_head_dim = 96`（非 2 的幂），`rocm_aiter_mla.py` 是否仍会走 `fused_mla_prefill_qkv_fp8`？
   - 验证：`96 & 95 = 64 != 0`，`not any(d & (d - 1) for d in head_dims)` 为 `False`，自动安全回退至 `_concat_k_nope_k_pe`。
