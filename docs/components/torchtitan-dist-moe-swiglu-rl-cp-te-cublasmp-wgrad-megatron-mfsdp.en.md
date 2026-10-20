# TorchTitan Dist-MoE MXFP8 Mega Fused SwiGLU Inference & RL Ulysses CP Per-Token Sharding, TransformerEngine cuBLASMp WGRAD Workspace Reuse & Fullgraph Linear Attention, and Megatron-FSDP Frozen Hook Deferral

> **Language / 语言**: [中文 (`torchtitan-dist-moe-swiglu-rl-cp-te-cublasmp-wgrad-megatron-mfsdp.md`)](torchtitan-dist-moe-swiglu-rl-cp-te-cublasmp-wgrad-megatron-mfsdp.md) | **English (Current)**
> **Evidence Class**: `implementation` + `author-benchmark` (upstream commits) + `own-offline` (deterministic stdlib simulator)
> **Upstream Anchors**:
> - `pytorch/torchtitan@ed3c713afeb21a514714ff59fec4fd4bb2d3b09d` (`2026-10-20`, PR `#5465`)
> - `pytorch/torchtitan@9371a2ffcb8655d74e6e1c099d7ca7b756259cae` (`2026-10-20`, PR `#5424`)
> - `NVIDIA/TransformerEngine@1aaede2f9edcaf70020c112c673e0b16818bc6a4` (`2026-10-20`, PR `#3629`)
> - `NVIDIA/TransformerEngine@5eeb62a2fcb020af9778e3d29b14b5dc5c5ad50a` (`2026-10-20`, PR `#3685`)
> - `NVIDIA/Megatron-LM@d37b3a8fdb37af1b3a1d75720636af4c323bd522` (`2026-10-19`, PR `#8113`)
> - `LMCache/LMCache@64b05b8af88d5fc13b375cdc06b5425cd56dea47` (`2026-10-19`, PR `#5791`)
> - `ai-dynamo/dynamo@2ae8a46299b1c7d0287d1ab8fe4b67018bafe560` (`2026-10-20`, PR `#16633`)

---

## 1. Architectural Findings & Root-Cause Analysis

### 1.1 TorchTitan Dist-MoE MXFP8 Mega Kernel with Fused SwiGLU for RL Rollout Inference (`ed3c713`)

In TorchTitan's RL rollout generator, Dist-MoE (`32x32` block-scaled MXFP8) previously executed the training-oriented **staged** pipeline (separate `W13` and `W2` kernels):
- **Mechanism (`ed3c713`)**:
  1. `DistMoeTransform` adds `mxfp8_mega_fused_inference=True` (requiring `expert_precision="mxfp8"` and `block_scaled_config.pipeline="mega"`).
  2. When `inference=True`, `_prepare_mxfp8_weight` allocates an empty tensor for `dgrad_scale` (`numel() == 0`, satisfying FSDP's non-`None` tensor field requirement while `prepared()` asserts `inference == (self.dgrad_scale.numel() == 0)`) and interleaves `W13` gate/up rows (`interleaved_fc13 = inference and gate_up`) so the Mega kernel fuses SwiGLU directly inside the `W13` epilogue within the same warp.
  3. **Numerical Parity**: All three pipelines (`staged`, `mega`, and `mega + fused swiglu`) match bitwise, with `0.066` normwise relative error against BF16 (`_MXFP8_REL_TOL = 0.12`, vs `~0.90` if gate/up rows are non-interleaved).
- **Author Microbenchmarks (`author-benchmark`, 1x B200, Qwen3-30B-A3B `EP=4` Dist-MoE layer)**:

| Routed Rows $R$ | Staged BF16 ($\mu\text{s}$) | Staged MXFP8 ($\mu\text{s}$) | Staged Speedup vs BF16 | Mega BF16 ($\mu\text{s}$) | Mega + Fused SwiGLU MXFP8 ($\mu\text{s}$) | Mega+SwiGLU Speedup vs BF16 | Mega+SwiGLU Speedup vs Staged MXFP8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | 105.2 | 117.3 | 0.90x | 105.6 | **61.3** | **1.72x** | **1.91x** |
| `2,048` | 122.9 | 125.6 | 0.98x | 123.8 | **72.0** | **1.72x** | **1.74x** |
| `5,120` | 144.8 | 151.1 | 0.96x | 144.2 | **87.4** | **1.65x** | **1.73x** |
| `8,192` | 174.1 | 171.4 | 1.02x | 174.0 | **112.0** | **1.55x** | **1.53x** |
| `10,240` | 218.4 | 192.9 | 1.13x | 219.8 | **135.5** | **1.62x** | **1.42x** |
| `16,384` | 300.4 | 230.7 | 1.30x | 300.4 | **192.0** | **1.56x** | **1.20x** |
| `32,768` | 450.4 | 342.6 | 1.31x | 450.7 | **292.8** | **1.54x** | **1.17x** |
| `65,536` | 790.3 | 583.6 | 1.35x | 789.6 | **516.8** | **1.53x** | **1.13x** |

On an 8x B200 node running **Qwen3-30B-A3B DAPO-Math RL** (median $R \approx 9,600$), end-to-end step speedup improves from `1.06x` to **`1.10x`** (`1.09x` generator decode step speedup).

---

### 1.2 TorchTitan RL Trainer Ulysses Context Parallelism (`9371a2f`)

- **Root Cause**: Under `CP > 1`, `preprocess_inputs` shards `labels` along the packed token dimension, but per-token RL `loss_kwargs` (`advantages`, `generator_logprobs`, `loss_mask`, `temperature`) reached GRPO/DAPO losses unsharded.
- **Fix (`9371a2f`)**: Adds `shard_per_token_tensors` in `torchtitan/distributed/context_parallel.py` to slice each per-token tensor into its CP rank's contiguous `[T / CP]` shard while rejecting `context_parallel_load_balancer != None`. On 4x GB300 (`Qwen3-0.6B`, `CP=2 x TP=2` vs `FSDP=2 x TP=2`), relative loss difference is **`2.0e-7`** (`0.3046408296`) and peak active GPU memory drops from **`4.47 GiB` to `3.92 GiB` (`-12.3%`)**.

---

### 1.3 TransformerEngine cuBLASMp 0.11.0+ WGRAD Workspace Reuse (`1aaede2`) & Fullgraph Linear Attention (`5eeb62a`)

1. **Zero-Copy WGRAD Reuse of cuBLASMp All-Gathered `grad_output` (`1aaede2`)**:
   - Starting in **cuBLASMp 0.11.0** (`nvte_cublasmp_version() >= 1100`), `nvte_all_gather_gemm(..., transb=false)` leaves the gathered `grad_output` tensor $(N, K)$ at the beginning of `ctx->workspace`.
   - `get_cublasmp_all_gather_output()` views that workspace in `Linear`, `LayerNormLinear`, `LayerNormMLP (fc2)`, and fused `te.ops` backward (for unquantized and per-tensor FP8 when `requires_dgrad` and not `delay_wgrad_compute`), reducing backward All-Gathers per layer from **`1 -> 0` in row-parallel `Linear`** and **`2 -> 1` in `LayerNormMLP`**.
2. **Full-Graph Compilation of Linear Attention (`5eeb62a`)**:
   - Resolves `cudnn.linear_attention.ops` once inside `LinearAttention.__init__` into `self._op`, preventing TorchDynamo from tracing `importlib` during `forward` and enabling `torch.compile(fullgraph=True)` across GDN, GDN2, and DeltaProduct (GDP).

---

### 1.4 Megatron-FSDP Frozen Module Hook Deferral (`d37b3a8`), LMCache Local PD Skip (`64b05b8`), and Dynamo Dead-MPI Fast Exit (`2ae8a46`)

- **Megatron-FSDP (`d37b3a8`)**: When a frozen module (`_trainable_parameter_countdown.initial_value == 0`) receives `all(grad is None for grad in grad_input)`, PyTorch's full backward hook fires *before* internal backward finishes. Appending `(module, post_backward_hook)` to `context._delayed_post_backward_callbacks` defers weight cleanup until backward completion.
- **LMCache (`64b05b8`)**: `PDBackendAsync.batched_submit_put_task` returns early when `transfer_spec is None` (local requests), preventing leaked `mem_obj.ref_count_up()` increments (`ref_count: 2 -> 1`).
- **Dynamo TRT-LLM (`2ae8a46`)**: `exit_worker_on_fatal_error` checks `engine.get_health_check_fatal_error()`; when `EngineDeadError` is present after an MPI child death, it skips `engine.shutdown()` / `engine.cleanup()` so the worker exits immediately instead of deadlocking on IPC to the dead rank.

---

## 2. Deterministic Verification (`own-offline`)

- **Experiment Script**: [`experiments/torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp/run.py`](../../experiments/torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp/run.py)
- **Command**: `python experiments/torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp/run.py --check`
- **Verified Invariants**:
  - `dist_moe_mega_swiglu_1_72x_and_empty_dgrad = true`
  - `rl_cp_per_token_shard_exact_and_mem_saved = true`
  - `te_cublasmp_0_11_eliminates_linear_wgrad_ag = true`
  - `mfsdp_lmcache_dynamo_lifecycle_safe = true`
