# veRL Low-Precision QAT Rollout In-Place Weight Refit & Tail-Aware Top-K OPD Simulator

Deterministic Python 3.12 standard-library simulator reproducing zero-copy weight storage aliasing (`_mxfp4_staging_data`, `_replace_parameter_in_place`) for CUDA-Graph-safe MXFP4×MXFP8 and NVFP4 W4A4 routed-expert refit, alongside coarse-grained tail-bucket forward KL (`compute_tail_bucket_kl`, `compute_tail_aware_logit_gradient`) from `volcengine/verl` commits `8f016846e3010e225072a947270dc0c1b98c1cd0` and `5a8dd519ab32bfd36d2509a137f830ba79e89c54`.

## Run & Verify

```bash
python experiments/verl_fp4_refit_tail_kl_sim/run.py --check
```
