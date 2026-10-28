# Phase 124 Track C Experiment: Megatron-LM GDP FP32 Sigmoid / Batch Ramp / Refit, TorchTitan Compile / Generic Loss, veRL TPU v7x / ROCm DSV4 / VeOmni, Mooncake EFA / WeakPtr / NUMA / HCCS, LMCache & Dynamo

Deterministic verification of:
- `NVIDIA/Megatron-LM` commits `a296523510`, `d1eec51b85`, `4a55e6fee6`
- `pytorch/torchtitan` commits `0f9fd2bd64`, `740d5f755a`
- `volcengine/verl` commits `2ab3529203`, `6d04f398f1`, `b65da71c17`, `6f3f449e11`
- `kvcache-ai/Mooncake` commits `3a05eec5ef`, `9df448399b`, `a69bdd41cf`, `179ffb21c8`, `e70c09986d`, `5ef59b0c6e`
- `LMCache/LMCache` commits `58c2ace2c2`, `0108f5b634` & `ai-dynamo/dynamo` commits `cd3b977edb`, `08907b7ad7`

## Run
```bash
python experiments/megatron_gdp_fp32_bsize_refit_torchtitan_compile_loss_verl_tpu_rocm_veomni_mooncake_lmcache_dynamo_phase124/run.py
python experiments/megatron_gdp_fp32_bsize_refit_torchtitan_compile_loss_verl_tpu_rocm_veomni_mooncake_lmcache_dynamo_phase124/run.py --check
```
