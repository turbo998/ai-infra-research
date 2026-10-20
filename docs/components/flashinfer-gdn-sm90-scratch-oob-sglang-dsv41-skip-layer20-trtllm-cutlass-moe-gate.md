# FlashInfer SM90 GDN 预填充 Scratch 越界修复、SGLang AMD DSV4.1 Layer-20 有界重放跳过与 TensorRT-LLM CUTLASS MoE / MLA 准入门控

> **English Version**: [flashinfer-gdn-sm90-scratch-oob-sglang-dsv41-skip-layer20-trtllm-cutlass-moe-gate.en.md](flashinfer-gdn-sm90-scratch-oob-sglang-dsv41-skip-layer20-trtllm-cutlass-moe-gate.en.md)

## 1. 一手源码证据与核心定位

| 系统 / 仓库 | 精确 Commit / 日期 | 核心源码路径 | 证据等级 |
|---|---|---|---|
| **FlashInfer** (`flashinfer-ai/flashinfer`) | `afbe429cc26216343f00c17fc9aa23fa5d8449ed` (`2026-10-20`, PR `#6849`) | `flashinfer/gdn_prefill.py`, `tests/gdn/test_prefill_state_indices.py` | `implementation` + `author-benchmark` |
| **SGLang** (`sgl-project/sglang`) | `82ac83a8d0ab50ea7c9fe8e5b5934c2fa5a224bf` (`2026-10-20`, PR `#45339`) | `python/sglang/srt/models/deepseek_v4.py`, `deepseek_v4_platform_stage.py`, `deepseek_v4_backend_hip_radix.py` | `implementation` |
| **TensorRT-LLM** (`NVIDIA/TensorRT-LLM`) | `fe92fb923f12f98cfafc180bd9496126d2d9ef9e` (`2026-10-20`, PR `#19753`) & `4c0197d1e4410eb005748224ffd4c2ca5f0c7b44` (`2026-10-20`, PR `#20652`) | `cpp/tensorrt_llm/thop/moe/cutlass/moeOp.cpp`, `tensorrt_llm/_torch/moe/fused_moe/configurable_moe.py`, `tests/unittest/_torch/attention/test_cute_dsl_mla.py` | `implementation` |

---

## 2. FlashInfer SM90 GDN Prefill：只读状态池的 Scratch 越界写 (`#6849`)

### 2.1 问题机理与显存地址算术
在 Gated Delta Rule (`chunk_gated_delta_rule`) 预填充中，当调用方传入 `state_indices`、`output_final_state=False` 且 `output_state=None`（即从只读状态池 `initial_state` `[N_pool, H, V, K]` 读取初始状态、无需返回最终状态）时，SM90 (`chunk_gated_delta_rule_sm90`) DSL 内核**依然无条件**将每条序列的最终状态写回 `output_state[state_indices[i]]`。

修复前，SM90 调度器仅按紧凑批大小分配了 `[num_seqs, H, V, K]` 的 `float32` 临时缓冲区：
$$\text{Bytes}_{\text{slot}} = H \times V \times K \times 4 = 16 \times 128 \times 128 \times 4 = 1,048,576\text{ B} = 1\text{ MiB}$$
当 `num_seqs = 2`、`N_pool = 5`、`slots = [3, 0]` 时：
- 紧凑 Scratch 大小为 $2 \times 1\text{ MiB} = 2\text{ MiB}$；
- 序列 0 写入 `output_state[3]`（偏移区间 $[3\text{ MiB}, 4\text{ MiB})$），越过 $2\text{ MiB}$ 紧凑缓冲区末尾达 **$2\text{ MiB}$**（写入了 $1\text{ MiB}$ 非法数据）。
- 在开启 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 禁用 PyTorch 缓存分配器掩蔽后，H100 上的 `compute-sanitizer --tool memcheck` 捕获到 **2,176 次非法全局显存写 (invalid global writes)**。

### 2.2 双重边界修复
1. **无池降级**：若 `initial_state is None and output_state is None`，无状态池可索引，直接将 `state_indices = None` 丢弃，使内核安全写入紧凑 `[num_seqs, H, V, K]`。
2. **只读池对齐**：当 `state_indices is not None and initial_state is not None` 时，Scratch 形状扩展为 `initial_state.shape`（即 `[N_pool, H, V, K]` = $5\text{ MiB}$），彻底消除越界写（`0` 次非法写）。

---

## 3. SGLang AMD ROCm：DeepSeek-V4.1 Layer-20 有界重放跳过 (`#45339`)

### 3.1 Fused mHC 边界与 Layer-20 全量冗余计算
在 `--enable-decoder-swa-bounded-replay` 下，DeepSeek-V4.1 的 Decoder 层在预填充阶段只需对每个请求末尾 $W_{\text{tail}} = 128$ 个 Token 执行重计算。但第 20 层（首个 Decoder 层，`last_source = max(config.kv_source_layer_ids) = 20`）必须为全部 $N_{\text{full}} = B \times L_{\text{prompt}}$ 个 Token 写入 KV Cache。

在 AMD ROCm (`_is_hip`) 路径上，由于 `hc_boundary_fused` 将上一层 FFN 的 `mHC post` 延迟到当前层起始融合核中执行，旧实现无法把 Layer 20 的 KV 写入与后续计算拆开，只能设置 `tail_writes_source_kv = False`、`late_layer_start = 21`，导致 Layer 20 在全部 $N_{\text{full}}$ 个 Token 上完整运行了 Indexer、Attention 与 MoE。

### 3.2 `fused_boundary_attn_input` 拆分与计算节省
`82ac83a8` 在 `deepseek_v4_platform_stage.py` 中新增 `fused_boundary_attn_input(layer, state)`，显式执行挂起的 `make_amd_hc_boundary` 与 `_input_norm`，随后调用 `attn_backend.write_tail_source_kv` 为全部 $N_{\text{full}}$ 个 Token 写入 Layer 20 KV，并将 `late_layer_start` 提前至 **Layer 20**：
- 当 $B=4, L_{\text{prompt}}=4096, W_{\text{tail}}=128$ 时，Layer 20 的 Indexer/Attention/MoE Token 数从 $16,384$ 降至 $512$（**单层重计算削减 $96.875\%$**）；
- 跨 40 个 Decoder 层的总有效重计算 Token-Layer 数从 $16,384 + 39 \times 512 = 36,352$ 降至 $40 \times 512 = 20,480$（**总体削减 $43.662\%$**）。

---

## 4. TensorRT-LLM：CUTLASS MoE 兼容性门控与 MLA H=96 批大小下限 (`#19753`, `#20652`)

1. **OSS 与内部 CUTLASS MoE 构建分离 (`#19753`)**：
   - 当未开启 `USING_OSS_CUTLASS_MOE_GEMM` 时，构建 `moe_host_src` (`TLLM_MOE_NO_GEMM`) 并在 `validateInternalMoeInputs` 中前置校验 5 项硬性不变量：`unpadded_hidden_size == hidden_size`、`num_valid_tokens == num_rows`、`!enable_alltoall`、`!has_input_sf || swizzled_input_sf`、`!use_dynamic_fc2_scale`。
   - `ConfigurableMoE` 将 `allow_alltoall=self.backend.supports_alltoall` 传入通信自动选择，不支持 All-to-All 时自动回退至 `AllGather/ReduceScatter`，并在 `try ... finally` 中确保校验异常时调用 `self.comm.destroy()` 释放通信句柄。
2. **SM103 MLA $H=96$ FP8 单 Token 路由阈值 (`#20652`)**：
   - CuTe DSL MLA 在 $H=96$、单 Token (`seq_len_q = 1`) FP8 解码下的实测性能下限为 `_PERF_MIN_BATCH_FP8[(96, 1)] = 8`：`batch_size = 4` 时正确回退至 `trtllm_gen`，`batch_size >= 8` 或多 Token 验证 (`seq_len_q > 1`) 时路由至 `cute_dsl_mla`。

---

## 5. 可复现离线验证

```bash
python experiments/flashinfer_gdn_sm90_sglang_dsv41_trtllm_moe/run.py --check
```
