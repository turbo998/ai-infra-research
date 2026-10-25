# FlashInfer CAKE Norm-Combine Packed Reduction, W4A16 Split Fused MoE, MiniMax-H3 N-Split Tail, Frost FP8 Decode, and TileLang AST Span Mapping

> 中文版: [flashinfer-cake-norm-combine-w4a16-moe-minimax-nsplit-frost-fp8-tilelang.md](flashinfer-cake-norm-combine-w4a16-moe-minimax-nsplit-frost-fp8-tilelang.md)

## 1. Context and Core Architectural Problems

Deploying mixed-precision MoE and hybrid-attention workloads on Blackwell (`sm_100a` B200 / GB200, `sm_103a` B300 / GB300), workstation Blackwell (`SM120` RTX PRO 6000), and Hopper (`SM90`) exposes five tightly coupled kernel and compiler bottlenecks:

1. **Multi-GPU Residual Combine + RMSNorm Reduction Latency (`cake_fused_norm_combine`)**: When reducing `BF16` activations (`H=2560`, 2 norm tracks) across 8 NVLink peers, waiting on remote control words and barrier synchronization before issuing local gamma/residual loads creates pipeline bubbles at small token counts (`T=1..64`), while scalar `BF16 -> FP32` unpacking inflates instruction count at larger token counts (`T=1024..2048`).
2. **Batch Crossover Boundaries in Blackwell `BF16 x NVFP4` (W4A16) Split Fused MoE (`cake_fused_moe`)**: When expert weights use `NVFP4` (E2M1 with UE4M3 block-16 scales) and activations remain in `BF16`, split gated SwiGLU FC1 / FC2 kernels outperform monolithic CUTLASS at low decode token counts and large prefill batches, but exhibit a mid-batch regression around `T=128`.
3. **Partial Last-Wave Tail Waste in Persistent Block-Scaled GEMMs (`minimax_h3_out_proj_nvfp4` & `mm_bf16_fp4`)**: When the final persistent wave of the MiniMax-H3 attention output projection occupies at most half a wave, executing full-width `256 x 256` tiles wastes over half the tail wave's CTA cycles.
4. **FP8 Numerical Accumulation Boundaries in Low-Latency Grouped MoE Decode (`Grouped Frost`)**: Accumulating native FP8 Tensor Core partials across a full `K=128` TMA staging tile degrades BF16 preactivation boundaries or DeepSeek `128x128` block-FP8 fidelity at `T=4..8`, `top-k=10`.
5. **Signed 32-Bit Offset Overflow in Long-Context Varlen Attention & Host Enqueue Overhead (`minimax_h3` SM120 / cuDNN Warm Stream / TileLang AST)**: SM120 varlen kernels index `[padded_tokens, heads=56, 128]` with signed 32-bit offsets that overflow around `299.5K` tokens, while allocating `torch.cuda.Stream` wrappers on warm cuDNN attention calls and losing AST column spans during TileLang macro expansion hurt host launch latency and compiler diagnostics.

---

## 2. First-Hand Source Code Audit

### 2.1 `cake_fused_norm_combine`: Packed `BF16x2` Peer Sums and Early Stream Loads (`flashinfer@66a81f1bca98`, #7330)

Across the 4 token-count programs per architecture (`__launch_bounds__(320)`) on `sm_100a` and `sm_103a`:
- **Early Stream Loads Before Control Synchronization**: Immediately after `griddepcontrol.wait`, all 320 threads issue vectorized `ld.global.v8.b32` loads for `gamma0[8]`, `gamma1[8]`, and the first token chunk's `x` and `residual` before reading `workspace[24]` control words or executing `barrier.sync 1, 320`.
- **Packed `BF16x2` Peer Sums & One-Trip Pipelined Body**: Peer payloads across the 3-slot ring buffer (`control_current = word % 3`, `control_dirty = (word + 2) % 3`) are reduced via packed `__nv_bfloat162` additions and packed norm conversions (`160` `v8.b32` vectors per token).
- **Measured Latency Across `T=1..2048` (`H=2560`, 8 Peers, 2 Norm Tracks)**:
  - **8×B200 (`sm_100a`)**: `T=1`: `10.112 -> 9.184 us` (`1.101x`); `T=8`: `11.520 -> 10.528 us` (`1.094x`); `T=64`: `14.175 -> 12.928 us` (`1.096x`); `T=256`: `16.768 -> 15.776 us` (`1.063x`); `T=1024`: `29.792 -> 28.416 us` (`1.048x`); `T=2048`: `46.560 -> 42.881 us` (`1.086x`), achieving a **`1.0812x`** geometric mean speedup.
  - **8×B300 (`sm_103a`)**: `T=1`: `10.272 -> 9.280 us` (`1.107x`); `T=8`: `12.384 -> 11.392 us` (`1.087x`); `T=64`: `15.008 -> 13.792 us` (`1.088x`); `T=256`: `17.472 -> 16.576 us` (`1.054x`); `T=1024`: `30.688 -> 28.991 us` (`1.059x`); `T=2048`: `47.008 -> 43.872 us` (`1.071x`), achieving a **`1.0775x`** geometric mean speedup.

### 2.2 `cake_fused_moe`: `BF16 x NVFP4` W4A16 Split Fused MoE and Mid-Batch Refutation Boundary (`flashinfer@cf5c9a6a415b`, #7315)

`flashinfer` adds 10 `cake_fused_moe` W4A16 programs per architecture (`sm_100a` and `sm_103a`) behind `cutlass_fused_moe(backend="cake")`, pairing a 12-byte `CakeFastDivmod` descriptor with split gated SwiGLU FC1 and FC2 kernels:
- **B300 (`sm_103a`) Benchmarks and Crossover Analysis**:
  - **Preset D (`H=7168, I=2048, E=256, top-8, 1 GPU`)**: At `D1 (T=1)`, CAKE reaches `50.98 us` (s1) / `51.40 us` (s2) vs CUTLASS `53.73 us` (**`1.054x`**), whereas at `D128 (T=128)` CAKE takes `1027.63 us` vs CUTLASS `974.94 us` (**`0.949x`**).
  - **Preset G (`H=6144, I=2048, E=256, top-8, EP8`)**: CAKE wins at `G8 (T=8)` (`55.98 vs 58.63 us`, **`1.047x`**), `G32` (`1.007x`), `G64` (`1.010x`), `G512` (`225.70 vs 227.97 us`, `1.010x`), and `G4096` (`310.51 vs 317.56 us`, **`1.023x`**), while trailing at `G128 (T=128)` (`133.62 vs 123.31 us`, **`0.923x`**).

### 2.3 `minimax_h3_out_proj_nvfp4` N-Split Tail and `mm_bf16_fp4` M128 Pair (`flashinfer@b4125442a533`, #7314; `2866b59272d7`, #7311)

- **Bitwise-Identical Half-Width N-Split Tail (#7314)**: When $M \ge 32$, total waves stay within `48` on SM103 or `10` on SM100, and the partial last wave occupies $\le 0.5$ wave, `minimax_h3_out_proj_nvfp4` executes the tail tiles as **`256 x 128` half-width output-column units**. Across the 40-shape suite, the 20 tail-split shapes on GB300 (`sm_103a`) improve by **`1.003x–1.041x`** (geomean **`1.021x`**, e.g. `P=8, M=4816`: `82.2 -> 79.0 us`, `1.041x`), and the 7 tail-split shapes on GB200 (`sm_100a`) improve by **`1.017x–1.052x`** (geomean **`1.042x`**).
- **`mm_bf16_fp4` `tcgen05` M128 Pair PDL Kernels (#7311)**: Adds 2-CTA M128 pair programs with PDL for `768 x 2112 x 2048`, delivering **`1.015x`** native / **`1.029x`** tiled speedup on GB300 and **`1.018x`** / **`1.023x`** on B200.

### 2.4 Grouped Frost FP8 and DeepSeek Block-FP8 Low-Latency Decode (`flashinfer@d3d7c2488783`, #7212)

For `H=2048, I=512, top-k=10` at `T=4, 8`:
- **Hopper `SM90` & Blackwell `SM120` Candidates**: Adds `SM90` per-tensor E4M3 FP8 (`WGMMA group2/stages4/K128`), `SM120` per-tensor E4M3 FP8 (native warp FP8 MMA `group2`/`group4`, `stages2/K128`), and `SM90`/`SM120` DeepSeek `128x128` block-FP8 candidates.
- **`K64` Partial Accumulation Invariant**: Native FP8 Tensor Core partials span at most **`K64`** before promotion through separate **FP32 additions** ($128 / 64 = 2$ promotions per `K128` TMA tile), preserving the BF16 preactivation boundary and dynamic 128-element intermediate quantization without relaxing numerical tolerances.

### 2.5 SM120 Varlen 32-Bit Offset Guard, cuDNN Raw Stream Lookup, and TileLang AST Spans (`flashinfer@a02c8c5e6b57`, #7320; `85fcf535775b`, #7331; `tilelang@8f4501f2c3fd`, #3558)

- **SM120 `MiniMaxH3VarlenPlan` 32-Bit Offset Guard (#7320)**: Enforces `_MAX_ELEMENTS = (1 << 31) - 1 = 2,147,483,647` on `padded_tokens * heads * 128`, `num_segments * heads * 128`, and `num_tiles * 256 * heads`. For `heads=56, head_dim=128, _BLOCK_N=128`, the maximum safe padded token count is **`299,520` tokens** (`299,521` raises `ValueError`), preventing silent 32-bit sign overflow on GPU.
- **cuDNN Warm Attention Raw Stream Lookup (#7331)**: Replaces `torch.cuda.current_stream(device)` object construction with `_get_raw_stream(device_index)`, cutting B200 host enqueue time from `32.796 -> 31.154 us` (**`-5.00%`**) on prefill and `35.717 -> 32.706 us` (**`-8.43%`**) on decode.
- **TileLang Eager AST Source-Span Preservation (#3558)**: Preserves full `(lineno, col_offset, end_lineno, end_col_offset)` spans on substituted user AST nodes while collapsing template scaffolding to the statement's first line.

---

## 3. Reproducible Verification & Summary Matrix

```bash
python experiments/flashinfer_cake_norm_combine_w4a16_moe_minimax_nsplit_frost_fp8_tilelang/run.py --check
python -m unittest tests/test_phase75_experiments.py
```

| Subsystem / Kernel | Hardware / Arch | Configuration / Shape | Baseline / Before | Optimized / After | Speedup / Key Metric |
|---|---|---|---|---|---|
| `cake_fused_norm_combine` (#7330) | 8×B200 (`sm_100a`) | `BF16, H=2560, T=1..2048` | `10.112..46.560 us` | `9.184..42.881 us` | Peak **`1.101x`** (`T=1`), geomean **`1.0812x`** |
| `cake_fused_norm_combine` (#7330) | 8×B300 (`sm_103a`) | `BF16, H=2560, T=1..2048` | `10.272..47.008 us` | `9.280..43.872 us` | Peak **`1.107x`** (`T=1`), geomean **`1.0775x`** |
| `cake_fused_moe` W4A16 (#7315) | B300 (`sm_103a`) | Preset D (`T=1`) / Preset G (`T=8, 4096`) | `53.73 / 58.63 / 317.56 us` | `50.98 / 55.98 / 310.51 us` | **`1.054x` / `1.047x` / `1.023x`** (`0.949x/0.923x` @ `T=128`) |
| `minimax_h3_out_proj_nvfp4` (#7314) | GB300 (`sm_103a`) / GB200 (`sm_100a`) | 20 shapes / 7 shapes (`256x128` N-split tail) | Full `256x256` tail wave | Half-width `256x128` tail | GB300 **`1.021x`** (up to `1.041x`) / GB200 **`1.042x`** (up to `1.052x`) |
| Grouped Frost FP8 Decode (#7212) | `SM90` & `SM120` | `H=2048, I=512, top-k=10, T=4..8` | BF16 / Naive FP8 | `K64` partial + FP32 promote | **2** FP32 promotions per `K128` TMA tile |
| SM120 Varlen Offset & cuDNN Stream (#7320, #7331) | `SM120` / B200 | `heads=56, D=128` / Warm Attention | Silent overflow / `35.717 us` | Guard @ `299,520` / `32.706 us` | Zero silent overflow; decode host enqueue **`-8.43%`** |
