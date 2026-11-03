# Phase 181: Distributed Training, Collective Fabrics & KV Transport Deterministic Verification Suite

Validates the quantitative distributed training and fabric claims for Phase 181 (`priority: 190`):
- **Megatron-LM `#8361` (`631e146322af`)**: TE attention CUDA graphs with `dp_balanced` THD sequence packing (`288/288` output/gradient checks passed on 4x GB200 `PP2/VPP2/CP2` with fused CSA ratio-4 and HCA ratio-128).
- **Megatron-LM `#8137` (`01fa4164c8b5`), `#7972` (`e1afbb93db5c`), `#8823` (`a0ef64615cc3`)**: MInf token-level live request snapshotting for RL rollouts (`0` lost in-flight tokens), DSv4 C/H/W + MTP HybridModel FLOPs calculator (`48/48` CPU test cases), and `ScheduleNode` consumer-stream input storage recording (`0` stream races).
- **TransformerEngine `#3160` (`b021980343e3`) & `#3782` (`5c3f067c28b6`)**: Strided batched GEMM (`BatchedLinear`) in BF16/MXFP8 and `AdamCapturableFunctor` load/store ordering optimization (`0` redundant unscaled writes when `inv_scale == 1`).
- **PyTorch `#202454` (`fa6060d66f9d`) & ROCm Systems `#14146` (`b1a305f97a6a`), `#14694` (`34442d5ccd80`)**: Triton `rocSHMEM` device-side `alltoall` + `broadcast` (`2` collectives), RCCL Zen 2/3/4 CPUID (`23/25`) Rome-class topology restoration across `5` consumers, and ROCr `HSA_EXT_POINTER_TYPE_SVM (7)`.
- **Mooncake `#5026` (`e5ef0e427a49`) & `#4787` (`e8dda21b8fab`)**: Level Zero XPU IPC exporter `pidfd` exit cleanup (`0` leaked VRAM bytes) and `ShardAllocator` DFS LRU `std::string_view` key deduplication (`2 -> 1` copy/object, `50%` reduction).
- **LMCache (`#6256`, `#6208`) & NIXL (`#2142`, `#2560`)**: Non-blocking ZMQ `NOBLOCK` + `POLLOUT` client loop (`0` head-of-line freezes), retryable `PeerL2AdapterBase` teardown, overflow-safe `nixlBasicDesc::covers()`, and `loadRemoteSections` error backend disconnection.

## Run
```bash
python experiments/megatron_thd_cudagraph_minf_snapshot_dsv4_flops_te_mxfp8_rocshmem_rccl_mooncake_nixl_phase181/run.py --check
```
