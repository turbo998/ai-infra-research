# Phase 182: Distributed Training, Collectives & KV Fabric Deterministic Verification Suite

Validates the quantitative training, collective, and KV fabric claims for Phase 182 (`priority: 191`):
- **Megatron-LM `#7798` (`ebcf9d67221d`), `#8645` (`4a4064f0631c`) & `#8850` (`993bf83ec875`)**: `moe_training_matches_inference=True` bitwise RL train-gen log-prob parity (`0` ULPs mismatch) on 4x GB300 (`EP=4`), reaching `4.357x` per-token NVFP4 log-prob speedup (`23.53 -> 5.40 ms`), `2.897x` NVFP4 train-step speedup (`59.47 -> 20.53 ms`), `178 MiB` static memory savings per layer, and `1.422x` MXFP8 log-prob speedup, plus fused batch-invariant squared-ReLU + MXFP8 quantization (`0.07–0.13 ms/step` saved on GB200).
- **TransformerEngine `#3789` (`720b25258be3`), `#3691` (`2d65f35b662c`) & PyTorch `#202772` (`1933a0a9ee51`), `#202227` (`b31d68db9561`)**: `SplitAlongDim` zero-copy packed-QKV backward views (`0` zero-filled full-shape allocations), fullgraph compiled MoE routing, PyTorch Inductor wide-broadcast (`>= 64`) producer realization (`36.4 -> 16.8 us` = `2.167x` on B200 `gpt-oss` MoE down-projection epilogue), and AOTI C-shim CUDA Graph capture.
- **ROCm/rocm-systems RCCL `#14880` (`f59fd15cb79a`) & `#14759` (`3b0c0acddf20`)**: `gfx1250` (`MI455X`) DDA LL one-shot AllReduce L2 send prefetch and batched 8-line peer polling (`8` lines/wait), plus SPX ReduceScatter DDA LL128 (`224 MiB` `< 8` ranks, `7.5 MiB/rank` `>= 8` ranks) and DDA LL (`1.0 MiB/rank`) caps.
- **Mooncake `#3548` (`748af1a53dba`) & `#4042` (`4704c67f13ff`), LMCache `#4697` (`338d5fba394e`), & NIXL `#2561` (`e2cd0283d5df`) & `#2571` (`13c52d21bdc6`)**: Allocation-base HIP IPC registration with protocol-scoped (`3` protocols: `hip`, `rdma`, `tcp`) refcounted alias unregistration, Redis metadata `3 s` timeout + auto-reconnect, LMCache `RawBlock` in-flight CQ reservation (`0` overflow panics), UCX EFA device-gated loading, and NIXL `64`-slot FNV-1a stack-buffered NVTX name resolution (`-6` allocations/iter).

## Run
```bash
python experiments/megatron_moe_rl_parity_rccl_gfx1250_dda_mooncake_hip_ipc_phase182/run.py --check
```
