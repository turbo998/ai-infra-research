# LMCache pygrpc-native Mode, NIXL GPUNetIO Independent QP Progress & Immediate Lock Release, Megatron GTP Full-Graph Scratch Lifetime, TE Distributed-Weight FP32 wgrad, TorchTitan Qwen3-30B-A3B DAPO MXFP8 RL, verl HDO Pre-Resume Onload & QAT Spec, Mooncake Tenant Namespaces & Dynamo Engine-Owned Retune Deep Dive (Phase 95 / Priority 104)

> **Evidence Class**: Upstream source implementation (`implementation`) + author-reported B200 benchmark (`author-benchmark`, TorchTitan PR #5598) + deterministic Python 3.12 standard-library modeling (`own-offline`).

## 1. Core Architectural Questions & Primary Upstream Evidence

| Source ID | Repository & Commit | Key Implementation Path & Architectural Invariants |
|---|---|---|
| `lmcache-mp-pygrpc-native-server-mode-pr6031-2026` | `LMCache/LMCache` @ `a873f152f8` (PR #6031) | Adds `--grpc-server-mode native` (`pygrpc-native`) to LMCache MP server, receiving RPCs on Rust `tokio` threads and dispatching via a single Python dispatcher thread (`1` vs `64` blocked Python gRPC threads at 64 in-flight RPCs). |
| `lmcache-bench-request-owned-blocks-verify-pr5827-2026` | `LMCache/LMCache` @ `6b2b754654` (PR #5827) | Introduces immutable `RequestBlocks` / `GroupBlocks` shared across TP workers with local worker-buffer checksum verification in `lmcache bench server`. |
| `nixl-gpunetio-independent-qp-cpu-completion-pr2222-2026` | `ai-dynamo/nixl` @ `9e7800fc16` (PR #2222) | Progresses independent QPs via `docaSlotDesc` rings with CPU-local completion reporting and releases the SDK GPU lock immediately after SQ reservation (`0.15 us` vs `2.00 us`, a `13.33x` lock-hold reduction) before WQE preparation/submission. |
| `megatron-gtp-preserve-scratch-lifetime-capture-pr7780-2026` | `NVIDIA/Megatron-LM` @ `ed9c2c8c1c` (PR #7780) | Preserves symmetric-memory scratch lifetime during `_FULL_ITERATION` CUDA Graph capture by registering internal `ag_event`, `_recompute_ag_event`, and `rs_event` per GTP chain and calling `symmetric_wgrad_pool.reset_completed_reuse_events()`. |
| `te-grouped-wgrad-main-grad-dtype-dist-weights-pr3397-2026` | `NVIDIA/TransformerEngine` @ `e32e1238b0` (PR #3397) | Types grouped-linear distributed-weight `wgrad` from `w.main_grad.dtype` (FP32) via `weight_grad_dtype` rather than `compute_dtype` (BF16), preventing pre-reduce-scatter BF16 rounding loss (`8.004` vs `8.0` across 8 ranks). |
| `torchtitan-qwen3-30b-a3b-dapo-math-rl-config-pr5598-2026` | `pytorch/torchtitan` @ `df4e4dc22a` (PR #5598) | Adds Qwen3-30B-A3B DAPO math 8K RL recipes on 4×B200 trainer + 4×B200 generator (DP4/TP1/EP4); `mxfp8_mega_fused_inference=True` cuts step time from `24.3 s` to `22.7 s` (`1.07x`), rollout wait from `15.8 s` to `13.6 s` (`1.16x`), and decode time from `20.2 s` to `17.8 s` (`1.14x`). |
| `verl-megatron-onload-optimizer-before-ckpt-resume-pr8235-2026` | `volcengine/verl` @ `0bd668c1aa` (PR #8235) | Onloads offloaded Megatron optimizers before `load_checkpoint` so `HybridDeviceOptimizer` (`optimizer_cpu_offload=True`) sees `is_cuda=True` shards during `load_state_dict`, preventing first-step `KeyError` / device mismatch crashes. |
| `verl-megatron-preserve-bridge-specs-causal-mask-qat-pr8361-2026` | `volcengine/verl` @ `4f8f090af3` (PR #8361) | Preserves provider-specific `transformer_layer_spec` across 20+ Megatron-Bridge model families under QAT and sets `activation_key = kFp8Dynamic128Sym` on `CompressedTensorsW4A8Mxfp4MoEMethod`. |
| `mooncake-master-tenant-namespaces-evictor-quota-pr4887-2026` | `kvcache-ai/Mooncake` @ `45bbc5d16b` (PR #4887) | Cuts `MasterService` over from 1024 `MetadataShard` locks to per-tenant `metadata::NamespaceTable` and per-key `ObjectRoute` locks, eliminating unrelated-key shard lock contention. |
| `dynamo-planner-engine-owned-retune-replay-router-pr17095-2026` | `ai-dynamo/dynamo` @ `dea508e1c1` (PR #17095) | Adds engine-owned retune with replay router control and `expand_domain` finite value-domain expansion to Dynamo Planner. |

## 2. Deterministic Verification

- **Command**: `python experiments/lmcache_pygrpc_nixl_gpunetio_megatron_gtp_te_wgrad_titan_dapo_verl_mooncake_dynamo_phase95/run.py --check`
