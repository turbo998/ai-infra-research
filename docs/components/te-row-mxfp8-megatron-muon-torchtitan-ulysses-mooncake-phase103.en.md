# TransformerEngine Row-Only MXFP8 Master-Weight Casts, Megatron-LM MFSDP v2 Muon Orthogonalization, TorchTitan Ulysses CP / Tied RoPE Cache & Mooncake TENT RDMA Fail-Stop (Phase 103)

- **Question ID**: `te-row-mxfp8-megatron-muon-torchtitan-ulysses-mooncake-phase103` (`priority: 112`, component: `distributed-training`)
- **Evidence Status**: `own-offline` (First-hand upstream commit inspection + deterministic Python 3.12 stdlib model `experiments/te_row_mxfp8_megatron_muon_mooncake_phase103/run.py`)
- **Chinese Companion**: [`te-row-mxfp8-megatron-muon-torchtitan-ulysses-mooncake-phase103.md`](te-row-mxfp8-megatron-muon-torchtitan-ulysses-mooncake-phase103.md)

## 1. Primary Upstream Sources (Zero-Overlap)

1. **TransformerEngine `3850986745` (#3488)**: `Support row-only MXFP8 distributed master-weight casts`
2. **TransformerEngine `0f8d92f5a7` (#3744)**: `[PyTorch] Set device during NCCL init in symmetric memory test`
3. **Megatron-LM `63f687dc83` (#8517)**: `Preserve GDP gate storage under chunkwise context parallelism`
4. **Megatron-LM `19895dafba` (#8512)**: `docs: Muon design for MFSDP v2`
5. **TorchTitan `b44774915d` (#5590)**: `[CP] Add Ulysses context parallelism for KDA and GDN`
6. **TorchTitan `e111f10d4d` (#5583)**: `[perf] Tied RoPE cache`
7. **LMCache `0395e677c4` (#5870)**: `[Token Dropping] [MVP 1/12] Define request-local token-drop state`
8. **Mooncake `1dd37f57ba` (#4383)**: `[Bugfix][TENT] Fail stop on terminal RDMA teardown errors`
9. **Mooncake `ddda75d351` (#4708)**: `[TENT][Bugfix] Refresh Local Rail Topology Before Slice Selection`
10. **verl `6b2b417a05` (#8315)**: `[tool] feat: rl-insight trace weight-sync stages and report trainer metrics by step`

---

## 2. Core Implementation Mechanics & Quantitative Analysis

### 2.1 TransformerEngine Row-Only MXFP8 Distributed Master-Weight Casts (`3850986745`) & NCCL Device Binding (`0f8d92f5a7`)

`mxfp8_scaling_compute_partial_amax_kernel` and `mxfp8_scaling_partial_cast_kernel` are templated on `bool kWithColumnwise`. Passing an empty `amax_colwise` / `output_colwise` tensor dispatches `kWithColumnwise = false`, skipping column-wise amax reductions, `e8m0_t` column scale loads, and column-wise FP8 writes. On a $16384 \times 6144$ master-weight shard (`192.0 MiB` BF16 read), FP8 write traffic drops from `192.0 MiB` to `96.0 MiB` (**50.0% write saving**) and total cast HBM I/O drops from `390.0 MiB` to `291.0 MiB` (**25.38% total I/O saving**). Commit `0f8d92f5a7` binds `device_id=torch.device("cuda", rank)` during symmetric memory NCCL initialization.

### 2.2 Megatron-LM MFSDP v2 Muon Distributed Orthogonalization (`19895dafba`) & GDP Chunkwise CP (`63f687dc83`)

Because Newton–Schulz orthogonalization in Muon requires whole logical matrices rather than row/block shards, `19895dafba` compares two MFSDP v2 strategies across the 5-parameter canonical example (`P0=(2,6), P1=(4,4), P2=(4,4), P3=(1,2), P4=(1,6)`, **52 elements** across **5 ranks**):
1. **`RowAtomic` / `BlockAtomic(32)` + `FsdpMuon` P2P Gather/Scatter**: Keeps equal 12-element rank segments (`60` elements with padding) for balanced forward/backward collectives. After reduce-scatter, `P0, P3, P4` are already whole on one rank (`0` optimizer traffic); only boundary-split matrices `P1` and `P2` gather remote fragments to a single compute owner and scatter updates back (**16 elements** moved total vs **128 elements** for boundary all-gather — an **8.0x reduction** with zero duplicate Newton–Schulz).
2. **`TensorAtomic` Whole-Matrix Ownership**: Assigns whole matrices to persistent owners (`[12, 16, 0, 16, 8]`, `52` elements, `0` padding), eliminating optimizer-step communication (`0` elements moved) at the cost of a **1.5385x** max-to-mean segment imbalance during forward/backward collectives.
3. **GDP Chunkwise CP (`63f687dc83`)**: Removes `.contiguous()` on `rearrange(z, "b l d -> l b d")` in `GatedDeltaProductMixer._postprocess` so `batch_size == 1` preserves projection storage layout.

### 2.3 TorchTitan Ulysses CP (`b44774915d`), Tied RoPE Cache (`e111f10d4d`), LMCache (`0395e677c4`), Mooncake TENT (`1dd37f57ba`, `ddda75d351`) & verl (`6b2b417a05`)

- **TorchTitan (`b44774915d`, `e111f10d4d`)**: Adds Ulysses head-sharded CP (`head_shard_degree = tp * cp = 8`) for KDA/GDN with local conv1d/gate slicing, and ties identical `RoPE` caches across layers in a `Decoder` stage (**80.0x reduction** on an 80-layer 128K model, `10,240.0 → 128.0 MiB`).
- **LMCache (`0395e677c4`)**: Introduces `LMCacheMPTokenDropRequestState` (`worker_batch_row = None` until worker assignment).
- **Mooncake TENT (`1dd37f57ba`, `ddda75d351`)**: Enforces `LOG(FATAL)` fail-stop when `RdmaContext::disable()` or `RdmaEndPoint::deconstruct()` fails with live QP/MR resources, and refreshes rail topology in `Workers::selectOptimalDevice` whenever `source.topo != rail.local()`.
- **verl (`6b2b417a05`)**: Traces `wake_up` and weight-sync rollout stages via `RLInsightLogger.trace_state_decorator` and logs step-indexed trainer metrics via `metric_log(name, value, step=)`.
