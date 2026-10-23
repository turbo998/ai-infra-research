# Deep Dive: SGLang Rust Server Abort Delivery & Multimodal Finalize, TensorRT-LLM SM100 QuACK SwiGLU Fusion & NVFP4 Sync Prefill Quantization Cap, and vLLM `cooperative_topk` Row-Length Sizing & CPU GDN Spec-Decode State Parity

- **Research Date**: 2026-10-24 (Phase 57, Priority 66)
- **Evidence Tier**: `implementation` + `author-benchmark` (source audit) + `own-offline` (deterministic stdlib verification)
- **Upstream Commits**:
  - `sgl-project/sglang@5a86d6ee17e1f2fd34cfa6d703ea38c581ab23fa` (PR `#45899`, 2026-04-20)
  - `sgl-project/sglang@eb8ef8084231078c24fa3d8dbfc8a74d0f8f90e4` (PR `#45888`, 2026-04-20)
  - `sgl-project/sglang@74c08f3dc5d98dddf052ba03122fc084a31a0269` (PR `#45914`, 2026-04-20)
  - `sgl-project/sglang@82ff16dad08809a8819fa02d975d1a44cf202041` / `3e5f5505ed` / `2fbd094a8c` (PR `#45979`, `#45987`, `#45995`, 2026-04-20)
  - `NVIDIA/TensorRT-LLM@90d4dcbb8ff49a64a24914b9e1eed0f74bed1d5d` (PR `#19773`, 2026-04-20)
  - `NVIDIA/TensorRT-LLM@2676d6ba0aa4e689ebecc92d6fd41e4b95c1aa22` (PR `#20779`, 2026-04-20)
  - `vllm-project/vllm@404c2b510e1d73df151d78e5576e1ee7ca0bcaa3` (PR `#62171`, 2026-04-20)
  - `vllm-project/vllm@f3871c3cbf8a608f7ee833ec4c2c5277f77bee41` (PR `#63339`, 2026-04-20)
  - `vllm-project/vllm@b2e178eaa94bcdd5f0f5e88dec7eb65df1587cc0` / `be08fa5355` (PR `#63190`, `#63360`, 2026-04-20)
- **Verification Script**: `experiments/sglang-rust-abort-mm-finalize-trtllm-swiglu-nvfp4-vllm-topk-gdn/run.py`

---

## 1. Architectural Motivation & Problem Statement

Across SGLang, TensorRT-LLM, and vLLM on 2026-04-20, three critical serving-engine issues were resolved:

1. **SGLang Rust Server Hung Aborts & Premature Multimodal Validation**: Under `--enable-rust-server`, scheduler-initiated aborts (`queued`, `preempted`, `retracted`) were written to the unconsumed `send_to_tokenizer` ZMQ socket instead of the Rust server, leaving client HTTP streams hanging indefinitely (`#45899`). Separately, Rust multimodal processors validated `input_ids` inside `process()` before caller-provided `image_hashes` were applied (`#45914`).
2. **TensorRT-LLM 128K Prefill `cudaFree` Synchronization Storm on GB300 (`#20779`) & SM100 BF16 SwiGLU Intermediate Traffic (`#19773`)**: Running NVFP4 MoE input quantization on an auxiliary CUDA stream (`#19960`) caused `record_stream` retention of multi-GB prefill tensors on DeepSeek-R1 FP4 (`isl=128K, PP4`), triggering `52` blocking `cudaFree` calls (`0.36 s`), inflating `cudaEventSynchronize` from `2.30 s` to `7.38 s`, stalling PP `SendRecv` for `2.98 s` vs `0.02 s`, and halving disaggregated throughput (`~460` vs `~893 tok/s`).
3. **vLLM `cooperative_topk` Stride Over-Sizing (`#62171`) & CPU GDN Speculative-Decode Conv-State Corruption (`#63339`)**: `cooperative_topk` sized per-CTA staging blocks using physical buffer `stride` (`256K..1M`) instead of actual row length `seq_len` (`40K`), forcing single-pass eligible rows into two-pass streaming. Meanwhile, CPU GDN linear attention truncated `conv_state` on partial speculative draft acceptance, causing a `66%` output mismatch.

---

## 2. Implementation Mechanics & Quantitative Analysis

### 2.1 TensorRT-LLM: `ASYNC_INPUT_QUANT_MAX_TOKENS = 1024` (`#20779`) & SM100 QuACK SwiGLU Fusion (`#19773`)

#### (a) Token-Count Gating for Auxiliary-Stream NVFP4 Input Quantization (`#20779`)
In `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py`:
- `ASYNC_INPUT_QUANT_MAX_TOKENS = 1024` gates `quantize_input_async` so decode batches (`<= 896` tokens on Kimi-K3) still overlap quantization on `aux_stream`, while prefill batches (`> 1024` tokens) quantize synchronously on the main stream without cross-stream `record_stream` retention:

| Metric (GB300 DeepSeek-R1 FP4 `isl=128K`, `PP4`) | Before `#19960` | Unfixed `main` | Fixed PR `#20779` | Impact (`author-benchmark`) |
|---|---:|---:|---:|---|
| 4-iter `cudaEventSynchronize` total | — | `7.38 s` (90 calls) | **`2.30 s` (8 calls)** | `-68.8%` event sync time |
| 4-iter blocking `cudaFree` calls / time | `0` | `52` calls (`0.36 s`) | **`0` calls (`0.00 s`)** | Eliminates allocator sync stalls |
| 4-iter PP `ncclSendRecv` stall time | — | `2.98 s` | **`0.02 s`** | `149x` reduction in PP bubble propagation |
| 3-run Context-Only Median TTFT (ms) | `1,200,228 / 1,198,861 / 1,204,054` | `1,691,192 / 1,706,087 / 1,598,272` | **`1,186,204 / 1,198,656 / 1,202,545`** | **`-29.12%`** median TTFT (`-32.9%` max TTFT) |
| Disaggregated Output Throughput (tok/s) | `~893` | `~460` | **`875.8`** | **`1.904x`** throughput recovery |

#### (b) SM100/SM103 QuACK BF16 Gate/Up GEMM + SwiGLU Epilogue Fusion (`#19773`)
In `tensorrt_llm/_torch/modules/gated_mlp.py` and `linear.py`:
- Calls QuACK `gemm_act(..., activation="swiglu", concat_layout=("B",), tuned=False)` on SM100/SM103 when `K % 8 == 0` and `I % 8 == 0`, eliminating the `[M, 2I]` BF16 write+read (`8 * M * I` bytes saved, or **80% of intermediate HBM traffic** — `256 MiB` per layer at `M=4096, I=8192`) and reducing BF16 roundings from `2` to `1`.

### 2.2 vLLM: `cooperative_topk` Row-Length Sizing (`#62171`) & CPU GDN Conv-State Parity (`#63339`)

- **Single-Pass Recovery in `cooperative_topk` (`#62171`)**: Computing `per_block = ceil(seq_len / CS)` instead of `ceil(stride / CS)` keeps a `40K`-token row on a 4-CTA cluster at `10,000 <= 49,152` elements/CTA (single-pass staging) even when `stride = 256K` (`65,536 > 49,152`) or `1M`. This yields up to **`8%` kernel speedup on B300** (`12.18 -> 11.24 us`) and **`2.8% / 2.5% / 1.5% / 1.5%` end-to-end speedup** on 4x GB200 GLM-5.3-NVFP4 AgentX at concurrency `1 / 4 / 8 / 16`.
- **CPU GDN Speculative-Decode Conv-State Layout (`#63339`)**: Preserves the full `state_len = conv_kernel + num_spec` window across uneven draft counts per row and checks `seq_lens[0] == state_len - (width - 2)`, eliminating a `66%` speculative-decode output mismatch.
- **`prompt_embeds` LoRA Passthrough Guard (`#63190`)**: Skips `modality == "prompt_embeds"` in `set_active_mm_loras`.

### 2.3 SGLang: Rust Abort Delivery (`#45899`, `#45888`), Multimodal `finalize` (`#45914`), and Diffusion Fixes (`#45979`, `#45987`, `#45995`)

- Routes scheduler-side aborts via `rust_server.push_abort(abort_req, req)` (`#45899`) and adds native Rust `POST /abort_request` (`#45888`).
- Runs `processor.finalize(&mut entry)` after `encode_items` and caller `image_hashes` overrides (`#45914`).
- Skips no-op `auto` residency dry-runs (`#45979`), serves `/v1/images` from in-memory bytes (`#45987`), and offsets LTX-2.3 BCG video RoPE by `video_sp_start` (`#45995`).

---

## 3. Architect Exercises & Falsifiable Questions

1. **Traffic Calculation**: Across 60 layers at `M=4096, I=8192` in BF16, how much HBM read/write traffic does QuACK fused SwiGLU save per forward pass?
   - **Answer**: $60 \times 8 \times 4096 \times 8192\text{ B} = 15.0\text{ GiB}$.
2. **Falsifiable Question**: Does raising `ASYNC_INPUT_QUANT_MAX_TOKENS` from `1024` to `2048` on GB300 reintroduce nonzero `cudaFree` calls under `PP4` 128K prefill?
