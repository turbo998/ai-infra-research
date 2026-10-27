# SGLang Qwen3.5/EAGLE Megakernel Binding & EPLB Multi-Replica Loader, vLLM DSv4.1 Shared-Stream Graph Capture & TRT-LLM KVCM V2 Confidential Computing Page Tables (Phase 114)

> **Evidence Classification**: `implementation` + `own-offline` (upstream commit analysis and reproducible CPU/offline model verification; not real GPU hardware profiling)

## 1. Upstream Code Paths & Runtime Invariants

| Evidence ID | Upstream Commit | Key Implementation Path & Runtime Invariant |
|---|---|---|
| `sglang-nvidia-megakernels-qwen35-eagle-binding-2026` | `sgl-project/sglang@8f1b7691ca` (`2026-10-27`) | Bumps `MEGAKERNEL_API_VERSION = 2` and adds `EagleBinding` (`eagle_binding.py`) for hybrid GatedDeltaNet + full-attention (`Qwen3.5`) target + EAGLE topk-1 draft chains while keeping all runners and CUDA graphs owned by SGLang. |
| `sglang-bcg-record-graph-breaks-without-running-capture-2026` | `sgl-project/sglang@bda3718df0` (`2026-10-27`) | Enters `enable_breakable_cuda_graph()` during warmup and capture so graph breaks are recorded without executing them during capture. |
| `sglang-jit-split-header-only-parallel-translation-units-2026` | `sgl-project/sglang@37e9bdb7a7` (`2026-10-27`) | Adds `cuda_units` in `python/sglang/kernels/jit/utils/compile/spec.py` to split header-only template instantiations across parallel Ninja translation units (`6.11x` build speedup across 8 units). |
| `sglang-eplb-load-all-local-replicas-per-expert-2026` | `sgl-project/sglang@e80120b4fb` (`2026-10-27`) | Populates all local physical replicas `[start_idx, end_idx)` for each logical expert via `_local_routed_expert_ids` while excluding out-of-range shared slots after elastic EP scale-up. |
| `vllm-dsv41-capture-decoder-replay-graphs-model-stream-2026` | `vllm-project/vllm@274a5e19a1` (`2026-10-27`) | Passes the model graph's `stream` into `graph_capture(device, stream)` when capturing DSv4.1 decoder replay graphs, sharing free blocks in the graph memory pool (`720 MiB` saved in our model). |
| `vllm-serial-avoid-redundant-tensor-ops-gil-2026` | `vllm-project/vllm@b4a0830f66` (`2026-10-27`) | Serializes contiguous CPU tensors via `TORCH_DTYPE_TO_NUMPY_DTYPE` (`.view(torch.uint8)` fallback for BF16), cutting GIL release/re-acquire transitions from `4` to `1` per tensor. |
| `vllm-minimax-m3-gb200-agentx-presets-tp4-tp8-2026` | `vllm-project/vllm@815df3af10` (`2026-10-27`) | Specifies GB200 AgentX NVFP4 presets for MiniMax-M3: 1 tray for TP4, 2 trays for TP8. |
| `trtllm-kvcm-v2-materialize-page-tables-under-cc-2026` | `NVIDIA/TensorRT-LLM@b542326c60` (`2026-10-27`) | Materializes KVCM V2 base-page rows on device under Confidential Computing via `gatherBasePageRows` + `copyBasePageRowsToDevice` when host tables are not directly GPU-readable. |
| `trtllm-refactor-child-request-preserve-inputs-2026` | `NVIDIA/TensorRT-LLM@0746a23338` (`2026-10-27`) | Deep-copies optional pointer inputs via `copyOptionalInput` in `LlmRequest(parent, requestId)` and commits child requests via `registerChildRequest`. |
| `trtllm-cute-dsl-gather-act-fusion-relu2-sf-midpoint-2026` | `NVIDIA/TensorRT-LLM@35cfb8cdbb` (`2026-10-27`) | Explains and bounds the ~`2.4%` E4M3 scale-factor midpoint divergence between FP32 kernel amax and BF16-rounded reference on unclamped `Relu2`. |

## 2. Verification Command

```bash
python experiments/sglang_megakernels_qwen35_eplb_vllm_dsv41_replay_trtllm_kvcm_phase114/run.py --check
```
