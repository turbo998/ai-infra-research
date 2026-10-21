# FlashInfer Masked MXFP8 Experts Quantize, SM90 MegaMoE Push Dispatch & CuTe-DSL `bmm_fp8` / `mm_mxfp8` One-Wave Priority

Deterministic standard-library verification of:
1. `flashinfer-ai/flashinfer` PR `#6437` (`140f9fc4c5`): SM100+ `cvt_fp16_to_mxfp8_expert` CUDA kernel and `silu_and_mul_mxfp8_experts_quantize` (`10.03x` GEMM1 and `18.69x` fused SiLU-and-mul speedups on B200).
2. `flashinfer-ai/flashinfer` PR `#6517` (`da992ed991`): SM90 FP4 MegaMoE inactive-warp `fetch_expert_recv_count()` workspace cleanup fix (`112 -> 0` dirty blocks at `128` hot-expert tokens), H200 EP8 push dispatch buckets (`M<=32` and `M=33..64` rotated pool), and `6 * scale <= 448` NVFP4 scale rebalancing.
3. `flashinfer-ai/flashinfer` PR `#6800` (`bf044ba86e`) & PR `#6848` (`617845d1a3`): Blackwell/Rubin (`sm_107`) `bmm_fp8` in-epilogue `a_scale[0] * b_scale[0]` product and `mm_mxfp8` one-wave output-tile priority (`N // 16 <= SM_count`, eliminating `63/150` B300 and `43/120` B200 PDL regressions).

## Usage

```bash
python experiments/flashinfer_masked_mxfp8_sm90_megamoe_bmm_fp8/run.py --check
```
