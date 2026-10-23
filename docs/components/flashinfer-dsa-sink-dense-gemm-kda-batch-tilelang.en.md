# FlashInfer Cake DSA Far-Sample Sink Scan, SM103a Dense Projection GEMM & FP32 Router Backward, KDA Document-Group Batching, and TileLang Analyzer Clone & Batched Finalize-Reducer Lowering

> **中文版**: [flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang.md](flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang.md)  
> **Phase 63 · Priority 72 · Track A (`question-202`)** · Verified Date: `2026-10-24` (UTC+08:00)

---

## 1. Core Architectural Problem & Context

In long-context mixed-precision training and Mixture-of-Experts (MoE) projection pipelines on Blackwell (`sm_100a` GB200, `sm_103a` GB300, `sm_107a`) and TileLang compiler lowering, four critical performance and correctness bottlenecks emerged:

1. **Whole-Unit Re-Stream Pathologies in DSA Indexer Under Attention-Sink + Local-Band Distributions**: `flashinfer.dsa_indexer.dsa_indexer_topk` uses **descending tile-order scan** on multi-segment rows for `sm_100a` and `sm_107a`. When token score distributions include both an initial attention sink ($j < 64$, additive boost $+5$) and a local band near the query ($q - j < 512$, additive boost $+3$), sampling the local band first inflates the initial filter threshold $\tau_0$. Once the descending scan reaches the attention sink at the start of the row, concentrated sink keys overflow the candidate pool (`pool_ovf`) and trigger a costly whole-unit re-stream from HBM.
2. **Missing `sm_103a` (GB300) Support, Small-$N$ Wave Quantization, and Missing FP32 Router Backward in `cake_dense_projection_gemm`**: Previously, `cake_dense_projection_gemm` compiled only `104` kernel instances (`47 sm_100a / 57 sm_107a`) with zero native `sm_103a` programs, suffered poor tile utilization for small $N \in \{32, 48, 64, 128\}$, and raised `NotImplementedError` on FP32 router backward (`dgrad` / `wgrad`).
3. **71.7 GiB Activation Checkpointing Arena Explosion in Long-Context KDA Training**: At $S=261,393$ and $H=64$, default KDA backward checkpointing allocates `4,488 B` per `(token, head)`, requiring **`71,703 MiB`** (`71.7 GiB`) for a single training arena.
4. **TileLang `tir::Analyzer` Variable-Binding Leakage and Truncated Batched 1D Reductions**: In `LowerTileOp`, sharing a single `arith::Analyzer` across `PartitionLoop` and `VectorizeRewriter` leaked temporary loop range bindings into subsequent `T.Parallel` loops, breaking vectorization; meanwhile, `FinalizeReducerLowerer` emitted only offset `0` when `layout_batch_size > effective_batch`, silently dropping remaining row reductions.

---

## 2. Upstream Implementation Paths & Quantitative Mechanics

### 2.1 FlashInfer Cake DSA Indexer: Far-Sample First Threshold & Non-FTZ PTX Helpers (`1ff9f67725` / `#7125`)

In `flashinfer/dsa_indexer/csrc/dsa_indexer_topk.cu`, commit `1ff9f67725` (`#7125`) redesigns the descending-tile-order scan (`__CUDA_ARCH__ != 1030`) around a **Far-Sample First Threshold**:
- **Ranking Only Far-Sample Keys for $\tau_0$**: The initial sample pass skips the last few local-window tiles and ranks only far-sample keys to set the initial threshold $\tau_0$. Those skipped local tiles are then scanned during the main scan alongside the row-head attention sink, eliminating candidate pool overflow (`pool_ovf`) and whole-unit re-streaming.
- **Shared Memory Reduction & Non-FTZ PTX Intrinsics**: `SMEM_TOTAL` on narrow/split programs drops by **12,160 B** (`207,104 -> 194,944 B`), and `dsa_indexer_common.cuh` adds `add_f32x2_noftz` and `fma_f32x2_rn_noftz` (`add.rn.f32x2` / `fma.rn.f32x2` without `.ftz`) to preserve IEEE-754 subnormal accuracy.

| Benchmark Row | Shape Description ($T, T_{kv}$, segments, top-$k$) | GB200 Prev (ms) | GB200 `#7125` (ms) | GB200 Speedup | GB300 (`sm_103a` Ascending Control) |
|---|---|---:|---:|---:|---|
| `Z1` | 262k doc, CP16 zigzag rank 0 ($16384, 270336, 2, 2048$) | `22.588` | **`12.692`** | **1.78x** (`1.7797x`) | `11.954 -> 11.825 ms` (`1.01x`) |
| `Z1k4096` | `Z1` with $\text{top-}k = 4096$ | `23.654` | **`13.721`** | **1.72x** (`1.7239x`) | `13.489 -> 13.346 ms` (`1.01x`) |
| `Z1k256` | `Z1` with $\text{top-}k = 256$ | `22.138` | **`12.561`** | **1.76x** (`1.7624x`) | `11.797 -> 11.676 ms` (`1.01x`) |
| `S6sink` | $1 \times 262\text{k}$ causal doc, sink stats ($262144, 262144, 1, 2048$) | `366.792` | **`211.554`** | **1.73x** (`1.7338x`) | `207.289 -> 208.151 ms` (`1.00x`) |
| `P1sink` | Recorded P1 geometry with sink stats ($16231, 268757, 8, 2048$) | `5.479` | **`3.450`** | **1.59x** (`1.5881x`) | `2.656 -> 2.671 ms` (`0.99x`) |
| `Z2cp` | $8 \times 32\text{k}$ docs, CP16 zigzag rank 0 ($16384, 270336, 16, 2048$) | `3.317` | **`2.252`** | **1.47x** (`1.4729x`) | `2.164 -> 2.138 ms` (`1.01x`) |
| `S4sink` | $1 \times 64\text{k}$ causal doc, sink stats ($65536, 65536, 1, 2048$) | `19.751` | **`14.671`** | **1.35x** (`1.3463x`) | `14.129 -> 14.184 ms` (`1.00x`) |
| `Z2` | $8 \times 32\text{k}$ docs, full causal ($262144, 262144, 8, 2048$) | `43.909` | **`34.133`** | **1.29x** (`1.2864x`) | `30.892 -> 30.858 ms` (`1.00x`) |

### 2.2 `cake_dense_projection_gemm`: `sm_103a` Programs, Small-$N$ Rules & FP32 Router Backward (`2706a076ac` / `#7108`)

1. **Program Fleet Expansion & Source Folding**: Compiled program instances expand from `104` (`47 sm_100a / 57 sm_107a`) to **`247`** (`80 sm_100a / 80 sm_103a / 87 sm_107a`). Companion commits `#7124` (`c01aa7779b`) and `#7122` (`6b38488084`) fold FP32 grouped GEMM `wgrad` onto its BF16 twin via `-DWGRAD_OUT=32` (keeping `5` physical `.cu` files) and drop 3 unreachable `c{1,2,4}_t8` Blackwell softmax kernels.
2. **Small-$N$ & Wave-Quantization Rules (`EXACT_WAVE_BLOCK_N = 224`)**: Adds narrow-tile dispatch for $N \in \{32, 48, 64, 128\}$ and wave-aligned `BLOCK_N = 224`, achieving **1.878x** speedup on GB200 `4096->512` FP32 `wgrad` (`91.0 -> 48.5 us`) and **3.548x** on GB300 `5120->512` FP32 `dgrad` (`261.1 -> 73.6 us` vs cuBLAS BF16).
3. **FP32 Router Split-BF16x3 Backward**: Configures `_ROUTER_BLOCK_K = 32, _ROUTER_BLOCK_N = 128` and `ROUTER_SPLITS = {"kk": 4, "kn": 1, "nn_t": 6}` (cutting transposed router `wgrad` terms from `11` to **`6`**) with fused `_router_split_bf16_pair_t`, delivering **1.943x** on GB200 `router 5120->256 dgrad` (`462.4 -> 238.0 us`) and **2.173x** on GB300 `router 6144->256 fwd` (`488.5 -> 224.8 us` vs `cuBLAS FP32 (BF16x9)`).

### 2.3 Cake KDA Training: Document-Group Batching (`fd5e515c6f` / `#7102`)

Commit `fd5e515c6f` (`#7102`) adds `batch_tokens=B` and `max_docs_per_batch=M` to `kda_train_arena` and `kda_train`:
- **Greedy Order-Preserving Partition (`doc_batches`)**: Splits packed `cu_seqlens` documents into contiguous groups $(d_0, d_1)$ that share a single checkpoint arena sized to the largest group:
  $$\text{Arena}_{\text{grouped}} = \max_{g} \left(\sum_{d \in g} L_d\right) \times H \times \text{BytesPerTokenHead}_{\text{lean}}$$
- **Segmented Route Guard**: Refuses proportional segment rows when `num_docs * H <= 4` on the segmented route so small chain counts do not stall on long documents.

| Benchmark Row (GB300, $S=261,393, H=64$) | Grouping & Option | Default Arena (MiB) | Option Arena (MiB) | Arena Reduction | B3 Latency Delta |
|---|---|---:|---:|---:|---:|
| `P14_h64_s261393` (4 docs: `23496, 88510, 114974, 34413`) | `lean=True, tail_passes=4` (unbatched) | `71,703` | `45,054` | **1.59x** | `79.817 -> 79.746 ms` (`-0.1%`) |
| `P14_h64_s261393` | `lean=True, tail_passes=4`, **2 doc groups** | `71,703` | **`25,755`** | **2.78x** | `79.817 -> 84.809 ms` (`+6.3%`) |
| `P14_h64_s261393` | `lean=True, tail_passes=4`, **3 doc groups** | `71,703` | **`19,806`** | **3.62x** | `79.817 -> 86.083 ms` (`+7.9%`) |
| `EQ4_H64` (balanced $4 \times 65,348$ tokens) | `lean=True, tail_passes=4`, **2 doc groups** | `71,703` | **`22,538`** | **3.18x** | `80.884 -> 81.286 ms` (`+0.5%`) |
| `EQ4_H64` (balanced $4 \times 65,348$ tokens) | `lean=True, tail_passes=4`, **1 doc/group (4 groups)** | `71,703` | **`11,280`** | **6.36x** | `80.884 -> 86.840 ms` (`+7.4%`) |

### 2.4 TileLang Compiler: Analyzer Clone Isolation & Batched Finalize-Reducer Loop (`fe8c34618f`, `741bf8b722`, `6db1fc771a`)

1. **`VectorizeRewriter` Analyzer Clone (`fe8c34618f` / `#3612`) & `PartitionLoop` Isolation (`741bf8b722` / `#3226`)**: Clones `analyzer->Clone()` before invoking `VectorizeRewriter` and constructs a local `arith::Analyzer` inside `LowerParallelLoop` so `PartitionLoop` variable bindings do not leak across sequential `T.Parallel` loops.
2. **Batched 1D Reduction Loop (`6db1fc771a` / `#3070`)**: Wraps `FinalizeReducerLowerer` emission in `for (size_t offset = 0; offset < layout_batch_size; offset += effective_batch)`, recovering all `layout_batch_size / effective_batch` reduction slices on 2D buffers.

---

## 3. Deterministic Verification & Architect Guidance

Verified by `experiments/flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang/run.py` and `tests/test_phase63_experiments.py`:
- Upgrade GB200 (`sm_100a`) long-context DSA workloads to `#7125` to eliminate 1.78x descending-scan slowdowns on attention-sink sequences.
- Use `kda_train_arena(lean=True, tail_passes=4, batch_tokens=...)` on 256k packed sequences to reclaim **45.9–60.4 GiB** (`2.78x–6.36x`) of HBM at `<8%` step time cost.
