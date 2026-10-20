# NCCL Device API (`nccl_device.h`), GPU-Initiated Networking (`ncclGin`) Symmetric Collectives & Fused `RMSNormGIN` Architecture

> **Language / 语言**: [中文 (Chinese)](nccl-device-gin-symmetric-collectives.md) | **English**

## 1. Architectural Motivation: Why `nccl_device` and `ncclGin` Symmetric Collectives?

In traditional host-initiated NCCL collectives, the Tensor-Parallel (TP) / Sequence-Parallel (SP) critical path (`ReduceScatter -> RMSNorm -> AllGather`) incurs three structural penalties:
1. **Multi-Kernel Launch & Host Proxy Handshakes**: Launching 3 separate CUDA kernels with 2 inter-kernel stream/proxy synchronizations adds **`41.5 us`** of fixed scheduling overhead;
2. **Redundant HBM Round-Trips**: Writing intermediate activations to HBM after `ReduceScatter` and reading/writing them again in `RMSNorm` before `AllGather` adds **`4 × shard_bytes`** of HBM traffic;
3. **Decoupled Intra-Node NVLS (`multimem`) and Inter-Node Rail-Ring RDMA**: Host-orchestrated collectives cannot specialize individual warps inside a single CTA to overlap inter-node RDMA PUTs with intra-node NVLS hardware multicast.

Grounded in first-hand source code from **`NVIDIA/nccl` (`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`)**—specifically `src/include/nccl_device/gin.h`, `src/device/symmetric/all_gather_gin.cuh`, `src/device/symmetric/reduce_scatter_gin.cuh`, and `docs/examples/07_kernel_fusion/03_rmsnorm_gin/c/main.cu`—this deep dive establishes the exact completion semantics, warp-specialization formulas, and fusion speedup boundaries of NCCL GPU-Initiated Networking (`ncclGin`).

---

## 2. First-Hand Source Code Mechanics (`implementation`)

### 2.1 `src/include/nccl_device/gin.h`: Strong vs. Weak Completion Signals & Segment Tags

In `src/include/nccl_device/gin.h` (`blob 24ba3855188d15ac47c2b107552b2b31e3c1273c`), `ncclGinSession::put` separates remote completion ordering into explicit strong and weak action structs:

- **Strong Signals (`ncclGin_StrongSignalInc`, `ncclGin_StrongVASignalInc`, `ncclGin_StrongSignalAdd`)**: Visibility of the signal update on the target peer guarantees that **all preceding PUTs** on that context have settled in destination memory;
- **Weak Signals (`ncclGin_WeakSignalInc`, `ncclGin_WeakVASignalInc`, `ncclGin_WeakSignalAdd`)**: Guarantees **only that the bundled PUT** has settled, permitting out-of-order completion across multi-rail/sprayed paths;
- **Weak Local Counter (`ncclGin_WeakCounterInc`) & `gin.flush(warps)`**: Guarantees local source buffer consumption so source slices in `window_send` / `window_recv` can be safely overwritten;
- **Physical `cuMem` Segment Tags (`ncclGinSegmentType_t`)**: `ncclGinSegmentTypeDevice = 0` (device HBM), `ncclGinSegmentTypeMixed = 1` (mixed `HOST_NUMA` + HBM), and `ncclGinSegmentTypeHostNuma = 2` (CPU-backed `HOST_NUMA`).

### 2.2 `src/device/symmetric/all_gather_gin.cuh`: Gray-Code Bank Diversification & Warp-Specialized Rail-Ring + NVLS `bcastMultimem`

In `ncclSymkRun_AllGather_RailRing_LsaSTMC` (`blob 8a39c5a3228535b8e5e9954ebde83942d5ebab11`), NCCL fuses inter-node Rail-Ring transfers with intra-node NVLS multicast:

```cpp
// Rotate by LSA rank with pattern 0 1 3 2 4 5 7 6 to diversify addresses.
uint32_t startOffsetMultiplier = lsaRank ^ ((lsaRank >> 1) & 1);
size_t startOffset =
  nElts > 0 ? ((size_t)startOffsetMultiplier * ncclSymkMcPerRankOffsetBytes % nElts) / chunkSize * chunkSize : 0;
```

1. **Gray-Code Address Rotation**: Across 8 intra-node `lsaRank`s (`0..7`), `lsaRank ^ ((lsaRank >> 1) & 1)` produces the permutation **`[0, 1, 3, 2, 4, 5, 7, 6]`**, staggering memory controller and NVSwitch bank accesses by `256 KiB` per rank;
2. **Asymmetric Warp Specialization**:
   - **Warp 0 (`threadIdx.x < WARP_SIZE`)**: Executes inter-node `gin.put(rail, nextPeer, ..., ncclGin_SignalInc{railSignals + rail.rank})` and `gin.waitSignal`;
   - **Warps `1..15` (`blockDim.x / WARP_SIZE - 1` warps)**: Wait on `railSignals + prevPeer` and broadcast received chunks across the local NVLink domain via `bcastMultimem`.

### 2.3 `src/device/symmetric/reduce_scatter_gin.cuh`: `rsAlgoHier` Two-Stage Warp Partitioning & Inbox Credit Bounds

In `rsAlgoHier` (`blob 00ced10cbd1945817c743ec638a221b59ca7555d`), after reserving 2 GIN control warps (`nWorkWarps = blockDim.x / 32 - 2`), NCCL partitions worker warps between Stage 0 (intra-node read/reduce + outbox post) and Stage 1 (inter-node rail inbox reduction) proportionally to read work:

$$\text{stage0\_work} = (1 \text{ if multimem else lsa.nRanks}) \times (\text{rail.nRanks} - 1), \quad \text{stage1\_work} = (1 \text{ if multimem else lsa.nRanks}) + \text{rail.nRanks} - 1$$

$$\text{stage0\_nWorkWarps} = \min\left(\text{round}\left(\frac{\text{nWorkWarps} \times \text{stage0\_work}}{\text{stage0\_work} + \text{stage1\_work}}\right),\ \text{nWorkWarps} - 1\right)$$

And `ncclGinInboxA2ASession` enforces credit-safe chunk bounds:

$$\text{minChunkBytes\_log2} = \text{ginInboxRail.size\_log2} - \min\left(\lfloor\log_2(\text{rail.nRanks}-1)\rfloor + \text{ncclGinScratchMaxBufsPerPeer\_log2},\ \text{ncclGinScratchMaxBufs\_log2}\right)$$

### 2.4 `docs/examples/07_kernel_fusion/03_rmsnorm_gin/c/main.cu`: Single-Kernel 3-Phase `RMSNormGIN`

In `RMSNormGIN` (`blob 7a50a31ff62f18a0ba1ef0b044314c210e0a2395`), one CTA per token executes **Reduce-Scatter (GIN PUT + `ncclGin_WeakSignalInc`) -> In-SM RMSNorm (in `reduction_buffer[]`) -> All-Gather (GIN PUT)** in a single kernel launch without resetting signals between phases.

---

## 3. Deterministic Offline Evaluation (`own-offline`)

Verified by [`experiments/nccl_gin_symmetric_sim/run.py`](../../experiments/nccl_gin_symmetric_sim/run.py) and [`results.json`](../../experiments/nccl_gin_symmetric_sim/results.json):

| Metric / Dimension | Unfused 3-Kernel / Non-Multimem Baseline | NCCL `ncclGin` Symmetric & Fused Path | Verified Architectural Gain |
|---|---|---|---|
| `rsAlgoHier` Worker Warps (`14` total) | `multimem=False`: **`stage0=11, stage1=3`** (`56:15` read work) | `multimem=True`: **`stage0=7, stage1=7`** (`7:8` balanced) | NVLS `multimem` eliminates `8x` intra-node read amplification |
| `AllGather_RailRing_LsaSTMC` Offsets | Uniform offset `0` (8-way bank collision) | Gray-code **`[0, 1, 3, 2, 4, 5, 7, 6]`** | **8 unique offsets** (`256 KiB` stride), `1` GIN warp + `15` Multimem warps |
| `ncclGinInboxA2ASession` Chunk Window | Unbounded small chunks exhaust credits | **`[256 KiB, 2 MiB]`** (`4 MiB` inbox) | Credit-safe pipelining across `rail.nRanks = 8` |
| Distributed `RS + RMSNorm + AG` (`T=4096, H=4096`) | 3 launches + 2 proxy syncs: **`234.57 us`**, **`80.0 MiB`** HBM | 1-kernel `RMSNormGIN`: **`110.92 us`**, **`64.0 MiB`** HBM | **`2.11x`** lower latency, **`6.38x`** lower launch overhead, **`-20.0%`** HBM traffic |

---

## 4. Architect Decision Guidelines & Falsifiable Questions

1. **When to adopt `RMSNormGIN` / `nccl_device` fusion**: Use for latency-critical TP/SP norm + collective stages where symmetric windows (`ncclCommWindowRegister`) are registered and per-step payloads are `256 KiB – 32 MiB`.
2. **Falsifiable Follow-Up**: Under multi-rail Packet Spraying with >16-packet reordering, what is the exact completion tail-latency gap between `ncclGin_WeakSignalInc` and `ncclGin_StrongSignalInc`?
