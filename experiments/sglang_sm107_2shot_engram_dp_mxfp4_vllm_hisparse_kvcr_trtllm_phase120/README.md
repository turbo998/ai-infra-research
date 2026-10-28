# Phase 120 Track B Experiment: SGLang SM107 2-Shot AllReduce, Engram DP Auto-Collective, Split-Expert MXFP4 Marlin, vLLM Spec-LoRA, HiSparse State-Slot & KVCR Prefix Recency

Reproducible Python standard-library model (`run.py`) and sealed output (`summary.json`) covering 10 upstream commits from `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:

- `sgl-project/sglang@08b9ec6`: SM107 (Rubin VR) custom all-reduce enables `two_shot_push = 16 MB` and lowers `1shot` limits on TP4/TP8/TP16 (`6.5013x` reduction on TP8).
- `sgl-project/sglang@e35b5bc`: DeepSeek-V4.1 Engram DP lookup `auto` collective (`all_to_all` moves `1 / tp_size` bytes when 24 hash columns divide evenly across `tp_size`).
- `sgl-project/sglang@faa56e2`, `bb4bb01`, `2eab19a`: Triton 3.9 `swap_ab` scalar scale broadcast, split-expert MXFP4 MoE via FP4 Marlin (`kFE8M0fnu`), and DSV4 draft metadata glue host-lengths guard.
- `vllm-project/vllm@b71c1d5`, `06e172b`, `66f1d0f`, `e3883b7`: Kimi-K3 ROCm folded weight hook, MRV2 speculative decoding `lm_head` LoRA logits alignment, HiSparse `[resident_groups, max_num_reqs, max_num_pages]` state-slot indexing, and KVCR prefix recency alignment.
- `NVIDIA/TensorRT-LLM@f4bc8ee`: Re-enable B300 (SM103) TRTLLM-Gen block-scale MoE cases after route-map over-read fix.

## Usage

```bash
python experiments/sglang_sm107_2shot_engram_dp_mxfp4_vllm_hisparse_kvcr_trtllm_phase120/run.py --check
```
