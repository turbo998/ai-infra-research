# Megatron-LM Warmup/Backward Partial Activation Offload & Per-Group Optimizer Step, PyTorch Inductor Live Reduction Lane Epilogues, ROCm HRR Pinned Snapshot Lock, Mooncake EngramStore & DMA-BUF 4GiB Wrap Guard, LMCache & NIXL (Phase 180)

Deterministic offline verification suite for Phase 180 (`priority: 189`, Track 3: `distributed-training`).

## Audited Upstream Commits
- `NVIDIA/Megatron-LM` @ `90f4d5279b95` (#7575), `2fcef8359b97` (#8485), `e0a4bfed7f1e` (#7158): Warmup/backward partial fine-grained activation offload fix (`37.5%` activation HBM reduction, `0` queue mismatches), unified preset FSDP `ExecutionOrder` traversal (`100.0%` determinism), and per-parameter-group optimizer step tracking (`0` step skews)
- `pytorch/pytorch` @ `4dd518df6250` (#200466) & `3da914e342b9` (#200463): TorchInductor forwarding live reduction lanes into persistent fixed-lane epilogues (`1.22x` speedup) and FP32 accumulation for half-precision grouped reductions in nested stages (`0` bitcast compilation errors)
- `ROCm/rocm-systems` @ `9051cfb62f24` (#14346): HIP Runtime Replay (`HRR`) shared/exclusive `host_free_mu` lock preventing pinned host snapshot use-after-free races (`0` races)
- `kvcache-ai/Mooncake` @ `e33fe7b83c06` (#4701) & `03ff956700e8` (#5113): Unified `EngramStore` table API with local CPU and stream-ordered CUDA Graph lookup (`1.38x` lookup speedup) and Transfer Engine DMA-BUF $2^{32}$-aligned MR wrap hazard detection (`100.0%` detection)
- `LMCache/LMCache` @ `38ebdeac201d` (#4072) & `ai-dynamo/nixl` @ `328a3c195c27` (#2516): Transfer-context close guard against in-flight gather/scatter (`0` crashes) and NIXL Python/pybind11 empty descriptor container handling (`0` `IndexError`/UB faults)
