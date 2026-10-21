# SGLang Mooncake Failed-Session Exponential Probe, GLM-5.2 Side-Stream Join & TensorRT-LLM GLM-5.3 `glm_kpool` Sparse Attention

Deterministic standard-library verification of:
1. `sgl-project/sglang` PR `#45338` (`be65391abd`) & PR `#43482` (`2e7c9d851c`): Default-on Mooncake failed-session probe with exponential backoff (`30s -> 60s -> 120s -> 240s -> 300s` max) and `600s` idle pruning (`2880 -> 4` probes per dead session), plus NIXL UCX per-GPU `device_list` resolution.
2. `sgl-project/sglang` PR `#45349` (`f7b8c6fe40`) & PR `#44929` (`da7e3f1c12`): GLM-5.2 DSA side-stream join ordering for single-stream CUDA-graph replay and UMBP Direct Linker packed draft layer mappings.
3. `NVIDIA/TensorRT-LLM` PR `#20454` (`ef913d39b3`) & PR `#19953` (`4636b6311f`): GLM-5.3-Flash `glm_kpool` sparse attention (`_append_paged_state_kernel` single-launch decode append, 2D draft-verification same-pool skip in `_kpool_update_kernel`, and native TRTLLM-GEN FP8 decode routing up to `128` heads without `[tokens, topk, 512]` BF16 gather/dequant) plus Rubin `SM107` CUTLASS MXFP8 MoE support.

## Usage

```bash
python experiments/sglang_mooncake_probe_glm52_trtllm_glm53_sparse/run.py --check
```
