# SGLang Qwen3.5/EAGLE Megakernel & EPLB Replica Loader, vLLM DSv4.1 Stream-Shared Replay & TRT-LLM KVCM V2 Confidential Computing Page Tables (Phase 114)

Deterministic CPU/offline verification suite for Phase 114 Track B (`priority: 123`):
- `sgl-project/sglang@8f1b7691ca`, `bda3718df0`, `37e9bdb7a7`, `e80120b4fb`: `MEGAKERNEL_API_VERSION = 2` Qwen3.5 + EAGLE binding, BCG capture without executing graph breaks, parallel JIT `cuda_units` (`6.11x` compile speedup), and EPLB multi-replica loading bounded to `[start_idx, end_idx)`.
- `vllm-project/vllm@274a5e19a1`, `b4a0830f66`, `815df3af10`: DSv4.1 decoder replay graph capture on shared model graph stream (`720 MiB` graph pool savings), zero-copy contiguous CPU tensor serialization (`4 -> 1` GIL transition), and MiniMax-M3 GB200 AgentX TP4/TP8 tray mapping.
- `NVIDIA/TensorRT-LLM@b542326c60`, `0746a23338`, `35cfb8cdbb`: KVCM V2 page-table materialization under Confidential Computing (`gatherBasePageRows` + `copyBasePageRowsToDevice`), deep-copied optional inputs for child `LlmRequest`, and FP32 vs BF16 E4M3 midpoint SF analysis for Relu2 vs SwiGLU/SiTu.

```bash
python experiments/sglang_megakernels_qwen35_eplb_vllm_dsv41_replay_trtllm_kvcm_phase114/run.py --check
```
