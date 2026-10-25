# TensorRT-LLM GLM-5.2 Rubin (`SM107`) Staircase Low-Latency Decode & NVFP4_AWQ MegaMoE, vLLM Shared-FS Idempotent KV Offload, and SGLang Quantized Paged Experts

> **Question ID**: `trtllm-glm52-rubin-awq-megamoe-vllm-fs-offload-sglang-paged-experts` (Phase 76, `priority: 85`)  
> **Depends On**: `trtllm-qwen3vl-graph-gemma4-vllm-cublas-tail-sglang-msa-nvfp4`  
> **Evidence ID**: `ev-trtllm-glm52-rubin-awq-vllm-fs-sglang-paged-experts-2026`  
> **Reproducible Experiment**: `experiments/trtllm_glm52_rubin_staircase_nvfp4_awq_vllm_fs_sglang_paged_experts/run.py`  
> **Chinese Version**: [`trtllm-glm52-rubin-staircase-nvfp4-awq-vllm-fs-sglang-paged-experts.md`](trtllm-glm52-rubin-staircase-nvfp4-awq-vllm-fs-sglang-paged-experts.md)

---

## 1. Architectural Motivation & Bottlenecks

Across next-generation GPU targets (NVIDIA Rubin `SM107` and Blackwell `SM100`), multi-instance shared-filesystem KV offloading, and host-to-GPU demand-paged MoE execution, four production challenges emerged in late October 2026:

1. **GLM-5.2 (`GlmMoeDsa`) Low-Latency Staircase Speculative Decode on Rubin (`SM107`, TP16)**: At small decode batch sizes ($M \le 8$), the first three dense MLP layers (`H=6144`, per-rank `I=768` under TP16) are strictly memory-bandwidth-bound (`27 MiB` BF16 weights/rank) and launch-bound across separate Gate+Up, SiLU, and Down kernels. Meanwhile, staircase MTP drafting requires local argmax, TP16 allgather, and slot updates on every speculative hop, and DeepGEMM lacked architecture family resolution for `SM 10.7` (Rubin).
2. **`NVFP4_AWQ` / `NVFP4_ARC` Pre-Quantization Scaling in MegaMoE CuteDSL**: Models such as Kimi-K2.5 rely on per-channel AWQ activation smoothing (`fc31_act_scale` / `w1_pre_quant_scale` / `w3_pre_quant_scale`) before NVFP4 quantization; omitting this step saturates `E2M1` ($\pm 6.0$) on outlier channels.
3. **vLLM Shared-Filesystem `EEXIST` Race on Concurrent KV Block Offloading**: When multiple vLLM workers write identical content-addressed KV blocks to a shared filesystem via `rename(tmp_path, dest_path)`, filesystems that return `EEXIST` (`FileExistsError`) when `dest_path` already exists caused losing workers to fail the offload job and leak `.tmp` files.
4. **SGLang Demand-Paged Experts (`--enable-paged-experts`) Quantization Coverage & Sub-Word Scale Alignment**: `paged_experts` previously supported only unquantized, block-FP8, and GPTQ formats, and `paged_experts_gather_kernel` required `row_bytes % 4 == 0`, crashing on 2-byte-aligned sub-word scale tables (e.g., 22-byte scale rows). Additionally, hybrid Mamba/GDN models using HiCache lacked host Mamba state-pool occupancy metrics, hiding silent prefix invalidation when the host Mamba pool filled before the host KV pool.

---

## 2. First-Hand Upstream Commit Audit

| Repository & Commit | Date | Evidence Tier | Key Source Files & Architectural Changes |
|---|---|---|---|
| `NVIDIA/TensorRT-LLM@e707e62fc4d3` (PR `#20517`) | `2026-10-24` | `implementation` | `cpp/tensorrt_llm/kernels/glm52Kernels/glmDenseMlpSmallM.{h,cu}`, `glmDraftSampleGlue.{h,cu}`, `glmDsaHistTopK.{h,cu}`, `jit_utils.py`: adds weight-streaming BF16 `glmDenseMlpSmallM` (`M <= 8`, swapped MMA operands on `m16n8k16`), 16B Lamport `glmDraftSampleGlue` (`kNone`/`kIncrement`/`kRewind`), histogram TopK, and DeepGEMM `10.x (minor > 3)` -> `100f` routing for Rubin `SM107`. |
| `NVIDIA/TensorRT-LLM@ecae3c97e2b6` (PR `#20823`) & `9666d7427a8c` (PR `#20774`) | `2026-10-25` | `implementation` | `mega_moe_cute_dsl.py`, `fused_moe_cute_dsl.py`, `w4a16_nvfp4.py`, `modeling_kimi_linear.py`: multiplies `x = x * fc31_act_scale` before FP4 activation quantization (and in W4A16 paths) and stages `w1/w3_pre_quant_scale` in streaming expert loads; `loraManager.cpp` preserves per-request LoRA format on reload. |
| `vllm-project/vllm@1757f6720c3c` (PR `#61390`), `503e4e16ad8d` (`#63126`) & `4a81a5097631` (`#55576`) | `2026-10-25` | `implementation` | `csrc/kv_offload/fs_io.cpp` + `vllm/v1/kv_offload/tiering/fs/io.py`: treats `EEXIST` / `FileExistsError` on `rename(tmp, dest)` as idempotent success by unlinking `tmp_path` and returning `0`; catches async lookup errors (`#63126`) and enables NIXL `OBJ` tier telemetry (`#55576`). |
| `sgl-project/sglang@2d3de82633b0` (PR `#45787`), `26a1e5110d4a` (`#43727`) & `97220dd67437` (`#46173`) | `2026-10-25` | `implementation` | `python/sglang/srt/layers/moe/paged_experts/` + `sgl-kernel/csrc/expert_specialization/paged_experts.cu`: expands `paged_experts` to 12 quantization formats and updates `paged_experts_gather_kernel` to select `16B` / `4B` / `2B` copies from even `row_bytes`; exports `sglang:hicache_host_mamba_used_states` and `total_states` (`#43727`). |

---

## 3. Formulation & Quantitative Analysis

### 3.1 TensorRT-LLM GLM-5.2 Rubin (`SM107`) Swapped-Operand Small-M MLP & Lamport Glue
- **`glmDenseMlpSmallM` ($M \le 8, H = 6144, I_{\text{local}} = 768$)**: Places weight rows on the `M=16` side and $\le 8$ token rows on the `N=8` side of `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` across $64 \times 4 = 256$ CTAs (`TILE_N=12`, `SPLIT_K=4`), streaming `27 MiB` of BF16 weights (`18 MiB` Gate+Up + `9 MiB` Down) in a single persistent grid launch.
- **`glmDraftSampleGlue`**: Packs `(bestVal, bestIdx, epoch, negZeroTag)` into a 16-byte Lamport packet across TP16 peers and fuses token selection with `kNone` / `kIncrement` / `kRewind` position updates.

### 3.2 vLLM Idempotent Shared-FS Rename & SGLang Sub-Word Gather
- **vLLM `fs_io.cpp` (`#61390`)**: Across 8 concurrent workers writing 64 shared KV blocks (`512` writes), handling `EEXIST` by unlinking `tmp_path` eliminates all `448` false write errors and `448` orphaned `.tmp` files.
- **SGLang `paged_experts_gather_kernel` (`#45787`)**: Selecting `16B` (`row_bytes % 16 == 0`), `4B` (`row_bytes % 4 == 0`), or `2B` (`row_bytes % 2 == 0`) enables sub-word scale rows (such as 22-byte scale rows) across all 12 quantized MoE formats (`AWQ`, `MXFP4`, `NVFP4`, `compressed-tensors`, `W8A8`, `block-INT8`, etc.).
