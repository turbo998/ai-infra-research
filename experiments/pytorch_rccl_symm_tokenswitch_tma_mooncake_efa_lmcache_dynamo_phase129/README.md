# Phase 129 Track 3 Experiment: PyTorch ROCm NCCLSymmetricMemory, TokenSwitch `topk_weights` Grad & Inductor `KERNEL_CONFIGS` + Mooncake EFA 88 GiB HugePages + LMCache Concurrent TP Transfers + Dynamo DCGM & Session Affinity

Deterministic verification of:
- `pytorch/pytorch` (`b8c7d2c`, `c5c9a00`, `b8d6389`, `e5d366e`)
- `NVIDIA/Megatron-LM` (`73f17bc`)
- `kvcache-ai/Mooncake` (`e648c15`)
- `LMCache/LMCache` (`d67052b`)
- `ai-dynamo/dynamo` (`f9c22d7`, `c03f514`)

Run:
```bash
python experiments/pytorch_rccl_symm_tokenswitch_tma_mooncake_efa_lmcache_dynamo_phase129/run.py --check
```
