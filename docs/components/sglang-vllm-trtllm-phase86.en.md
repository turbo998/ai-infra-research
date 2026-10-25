# SGLang Oversized AllReduce Slicing & External Cache Linkers, vLLM MRV2 2D Staged Writes & RDNA3 W4A8, TensorRT-LLM Compact Stop Words & QuACK SwiGLU Fusion (Phase 86)

> **Language / 语言**: [中文](sglang-vllm-trtllm-phase86.md) | **English (Current)**

## 1. Background and Core Bottlenecks

In production LLM and video diffusion serving engines, **IPC handle leaks, implicit host-device synchronizations, serialized long-context page-table writes, and worst-case sampler buffer growth** directly bound steady-state throughput and tail latency. Phase 86 Track 2 audits 10 primary commits across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:

1. **SGLang Oversized Deterministic AllReduce Slicing, Diffusion Sync Elimination & Out-of-Tree Cache Linkers** ([PR #34543](https://github.com/sgl-project/sglang/commit/656331295619a4d0267b40b7da0b5cd5b549fcbc), [PR #46047](https://github.com/sgl-project/sglang/commit/8ef5dec95faaa9c1192b96438c9d36eb988cd2db), [PR #40595](https://github.com/sgl-project/sglang/commit/f82af9ca17c9a115fd9c1d168c4cc547d1933f93)).
2. **vLLM MRV2 2D Staged Block-Table Writes, Clef Metadata Coalescing, PCP Forward Bench, UMBP Hybrid Connector & RDNA3 MXFP4 W4A8 GEMV** ([PR #63613](https://github.com/vllm-project/vllm/commit/237d2ffd08044b52625453af59f1e47b9802d4ff), [PR #63495](https://github.com/vllm-project/vllm/commit/0765565f103d5579382ad61994a041654684778b), [PR #62698](https://github.com/vllm-project/vllm/commit/fc2b6ab45ed3e498b52d74fd880a8b40a65ba3a7), [PR #61791](https://github.com/vllm-project/vllm/commit/c0f9fff649e732253c5472d3a4d28389605b4a01), [PR #60413](https://github.com/vllm-project/vllm/commit/c0ba7e01ffdd446a087df759e8c8c2bda28de2ea)).
3. **TensorRT-LLM Per-Request Compact Stop-Word Allocations & SM100/SM103 QuACK BF16 Gate/Up SwiGLU Epilogue Fusion** ([PR #20679](https://github.com/NVIDIA/TensorRT-LLM/commit/b96ba434e8a05e2ab52d2082bd8d70b1d844739b), [PR #19773](https://github.com/NVIDIA/TensorRT-LLM/commit/90d4dcbb8ff49a64a24914b9e1eed0f74bed1d5d)).

---

## 2. First-Hand Source Code Analysis

### 2.1 SGLang: Oversized Deterministic AllReduce Slicing, Zero-Sync Diffusion & External Linkers

- **Stop Registering Oversized Inputs in AMD Deterministic Custom AllReduce (PR #34543)**:
  - In `custom_all_reduce.py`, when deterministic all-reduce received chunked-prefill activations exceeding `self.max_size` (8 MiB), calling `deterministic_all_reduce` invoked `self.register_buffer(inp)` on every temporary tensor—permanently leaking `rank_data` slots until crashing with `"Rank data buffer is overflowed"`.
  - `_deterministic_all_reduce_oversized(inp)` slices oversized inputs into $\text{chunk} = \lfloor \text{max\_size} / \text{elem\_size} \rfloor$ pieces ($4,194,304$ BF16 elements) and loops over `ops.deterministic_all_reduce_unreg(self._ptr, slice_in, slice_out)`. A `16384 x 4096` BF16 (`128 MiB`) tensor is processed in `16` unregistered slices with **`0` leaked IPC handles**.
- **Eliminate Per-Step Host Syncs and Uploads in Video Diffusion Models (PR #46047)**:
  - Replaces 0-dim tensor indexing `block_params[:, cache_plan_index]` with `index_select(0, cache_plan_index.view(1))` in `minimax_h3_adaln_cache.py`, caches stacked PDD heads on device (`_pdd_device_heads`) in `minimax_h3.py`, and replaces `if bool(invalid.any()):` host syncs with `torch.where` in `sana_wm_components.py`, reducing H2D uploads across a 50-step denoise loop from `300` to **`6` (50.0x reduction)**.
- **Out-of-Tree External Linkers for UnifiedRadixCache (PR #40595)**:
  - Adds `--unified-cache-external-linker-backend {mooncake,mori,dynamic}` and `--unified-cache-external-linker-config`, validating config in the frontend via `UnifiedCacheLinkerConfig.from_dict()` and dynamically instantiating `UnifiedCacheLinker` plugins per worker via `create_unified_cache_linker()`.

### 2.2 vLLM: MRV2 2D Staged Writes, Clef H2D Coalescing, UMBP Hybrid & RDNA3 W4A8

- **Parallelize Staged Block-Table Writes Across 1024-Element Chunks (PR #62698)**:
  - Tracks `self._staged_write_max_len` in `vllm/v1/worker/gpu/buffer_utils.py` and launches a 2D grid `_apply_write_kernel[(n, _num_write_chunks(self._staged_write_max_len))]` with `_WRITE_BLOCK_SIZE = 1024` ($64 \times 3 = 192$ programs for $64$ requests with up to $2500$ block IDs), eliminating per-request serial tail loops.
- **Reduce Clef Pooling Metadata Overhead (PR #63495)**:
  - Packs all 7 index vectors on CPU with NumPy in `clef_joint_schema.py` and performs a single `async_tensor_h2d(np.concatenate(vectors))` followed by device `.split()`, cutting H2D launches from `7` to **`1`**.
- **RDNA3/RDNA3.5 (`gfx11`) W4A8 `int8`-Dot MXFP4 GEMV (PR #60413)**:
  - In `csrc/rocm/mxfp4_w4a8_gemv_rdna3.cu` ($1 \le M \le 8$), quantizes activations to symmetric per-32-element `int8`, maps MXFP4 `E2M1` weight codes losslessly to `int8` ($2 \times \text{E2M1} \in \{0, \pm 1, \pm 2, \pm 3, \pm 4, \pm 6, \pm 8, \pm 12\}$), and reduces each 32-element group with **8 `v_dot4_i32_iu8`** instructions followed by a single FP32 scale multiply.
- **Additionally**: [PR #63613](https://github.com/vllm-project/vllm/commit/237d2ffd08044b52625453af59f1e47b9802d4ff) adds Prefill Context Parallelism (`PCP`) support to `vllm bench forward`; [PR #61791](https://github.com/vllm-project/vllm/commit/c0f9fff649e732253c5472d3a4d28389605b4a01) adds shared lookup, cancellation, hybrid FullAttention+Mamba support, and Prometheus metrics to the UMBP KV connector.

### 2.3 TensorRT-LLM: Compact Stop-Word Buffers & SM100 QuACK SwiGLU Epilogue Fusion

- **Avoid Global Stop-Word Buffer Growth in PyTorch Sampler (PR #20679)**:
  - Replaces the monotonically growing global `[max_num_stop_words, max_stop_word_length, max_num_sequences]` tensor in `FinishReasonsHandler` with a fixed $5 \times \text{int64}$ (`40 B`) per-slot descriptor in `stop_words_metadata_cuda` plus per-request compact `int32` allocations freed in `PyExecutor._free_request_resources` via `sampler.release_request(request)`. Under an outlier stop-word workload ($128 \times 256$ stop tokens, $1024$ max sequences, $4$ active requests), memory drops from `138.41 MiB` to **`42.45 KiB` (3338.8x reduction)**.
- **Fuse SwiGLU into BF16 Gate/Up GEMM Epilogue on SM100/SM103 (PR #19773)**:
  - Integrates QuACK `gemm_act` (`concat_layout=("B",)`, `tuned=False`) into `GatedMLP` (`gate_up_swiglu_quack.py`), applying $\text{silu}(\text{gate}) \odot \text{up}$ directly to FP32 accumulators in the SM100 GEMM epilogue and storing only `[M, I]` BF16—eliminating **`448.0 MiB`** of intermediate `[M, 2I]` HBM write+read traffic per FFN layer at $M=8192, I=14336$.

---

## 3. Reproducible Verification (`experiments/sglang_vllm_trtllm_phase86`)

Running `python experiments/sglang_vllm_trtllm_phase86/run.py --check` deterministically verifies:
- **SGLang**: `16` unregistered slices for `128 MiB` all-reduce (`0` leaked handles) and `50.0x` fewer diffusion H2D uploads;
- **vLLM**: `192` 2D block-table write programs, `7 -> 1` Clef H2D transfers, and `8` `v_dot4_i32_iu8` ops per 32-element group;
- **TensorRT-LLM**: `3338.8x` stop-word buffer compression and `448.0 MiB` saved intermediate HBM traffic.

## 4. Architect Exercises & Falsifiable Questions

1. **HBM Traffic Calculation**: For a BF16 `GatedMLP` with $M=8192, K=4096, I=14336$, compute the exact percentage of total HBM traffic saved by eliminating the `[M, 2I]` intermediate write and read.
2. **Falsifiable Hypothesis**: Why does TRT-LLM keep `fuse_bf16_gate_up_swiglu` opt-in (enabled on large-token models like MiniMax-H3) rather than unconditional at small decode batch sizes $M \le 32$?
