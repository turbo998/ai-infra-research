# TransformerEngine Blackwell FP8 Block-Scaling to MXFP8 Grouped GEMM Swizzle Simulator

Deterministic Python 3.12 standard-library simulator reproducing the grouped scale-swizzle kernel geometry (`grouped_swizzle_block_scaling_to_mxfp8_kernel`), dynamic shared-memory prefix-sum tables (`table_smem_bytes`), binary-search tile lookup (`find_tensor`), and host upper-bound buffer sizing (`grouped_mxfp8_scale_bytes_upper_bound`) from `NVIDIA/TransformerEngine` commit `f4a69dc81eabb5002a662e56a36e8454c9840a92`.

## Run & Verify

```bash
python experiments/te_grouped_mxfp8_swizzle_sim/run.py --check
```
