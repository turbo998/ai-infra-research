# SGLang GLM DSA/MTP KVShard & Non-Interleaved `fmha_v2`, vLLM ROCm Single-Launch DSA Mask, and TensorRT-LLM Qwen Vision RoPE & V2 Long-Prefill Demotion (Phase 105)

Deterministic standard-library verification for Phase 105 (`priority: 114`) Track B:
- **SGLang GLM DSA/MTP KVShard (`8136cbce3d`), Non-Interleaved `fmha_v2` (`1a120754e0`) & BF16 MoE Refit (`8353f86f0b`)**: Shards `708 B/tok` across 8 prefill CP/TP ranks (`88.5 B/tok/rank`, `8.0x` unique KV capacity), eliminates `32.0 MiB/layer` GQA prefill KV interleaving copies (`64 -> 0` launches), and restores `128x128` blocked BF16 expert weights before RL refit.
- **vLLM ROCm Single-Launch DSV4.1 DSA Candidate Mask (`228b1e3d23`), Hybrid Mamba Spec Slots (`e3ba8d00db`) & EAGLE Quantized Head Guard (`43111fd1f0`)**: Fuses ROCm DSA decode candidate mask with `s_waitcnt vmcnt(0)` (`2 -> 1` launch, `19.6 -> 10.4 us`, `1.8846x` speedup) and reserves `1 + K` Mamba recurrent state blocks (`1 -> 4`).
- **TensorRT-LLM Qwen One-Launch Vision RoPE (`55f7033f71`) & V2 Scheduler Long-Prefill Demotion (`27c339ab53`)**: Halves ViT RoPE launches (`64 -> 32`, `19.2 -> 7.8 us`, `2.4615x`) and slashes short-request P95 TTFT under long uncached prefill arrivals from `285.0 -> 42.0 ms` (`6.7857x`).

```bash
python experiments/sglang_kvshard_glm_fmha_vllm_rocm_dsa_trtllm_phase105/run.py --check
```
