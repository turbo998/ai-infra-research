# TE GroupedMLP Split-Alignment, Megatron SchedulerConfig & TorchTitan Llama-3.2 RoPE 32x, and NIXL EP 8x8 CPU Proxy & Mooncake Conductor Rust API

Deterministic Python 3.12 standard-library model reproducing:
- `NVIDIA/TransformerEngine@9d9bd9e5c4` (#3584) & `@33f3b35619` (#3659): `GroupedLinear(..., split_alignment=256)` (`128` for MXFP8/NVFP4/FP8 block scaling) automatic `GroupedMLP_CuTeGEMMGLU` fusion and 0-D `()` FP8 current-scaling fake-tensor metadata.
- `NVIDIA/Megatron-LM@e4eb89b3bb` (#8211), `pytorch/torchtitan@48bbce4434` (#5528), & `@efdde5edaf` (#5500): `SchedulerConfig` migration, Llama-3.2 1B/3B `scaling_factor=32.0` (`17/32` and `35/64` rotary frequencies fixed, `0.781 rad` angle error eliminated, HF logit diff `3.2e-4 -> 7e-7`), and per-module `_MissingKernel[<module>]` Attention Gym guards.
- `ai-dynamo/nixl@fda3a883c0` (#2328), `ai-dynamo/dynamo@8e26089d3f` (#16759), `LMCache/LMCache@18651e8f19` (#5950), & `kvcache-ai/Mooncake@a4fe83974b` (#4651): NIXL EP 8-channel/8-thread CPU proxy (`59.0 GB/s` / `103%` on H100, `53.3 GB/s` / `109%` on GB200), Dynamo visible stop-token forwarding (`[2, 7, 200002, 200012]`), LMCache `inprocess` transfer channels, and Mooncake `ConductorClient` C ABI / Rust bindings.

## Run
```bash
python experiments/te-grouped-mlp-align-megatron-sched-titan-rope32-nixl-proxy-mooncake/run.py --check
```
