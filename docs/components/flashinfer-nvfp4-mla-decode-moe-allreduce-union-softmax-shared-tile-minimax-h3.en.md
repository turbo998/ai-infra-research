# FlashInfer SM100/SM103 NVFP4 MLA Decode Native E2M1×E4M3 Dequant, Latency-Ordered One-Shot/Two-Shot MoE All-Reduce Union, Shared-Tile Cluster Softmax, and MiniMax-H3 256×448 MXFP8 FC1

## 1. Engineering Problem Definition & Architectural Payoff

In production Blackwell deployments (`sm_100a` B200 / `sm_103a` GB300) and workstation/consumer Blackwell (`sm_120a` RTX 5090 / RTX PRO 6000), decode tail latency and long-context throughput face four microarchitectural bottlenecks:

1. **DeepSeek-V4 NVFP4 MLA Decode Dequantization & Online Softmax Dependency Chain** (`flashinfer-ai/flashinfer@923ff1321520`, `#7306`): MLA decode serves MQA over a single 512-wide NVFP4 latent row acting as both K and V (`E2M1` values + `UE4M3` block-16 scales, 64-token pages). Prior kernels unpacked `E2M1` and `UE4M3` to FP32 before multiplying, lengthening register live ranges and the online softmax rescaling chain.
2. **One-Shot NVLink Read Amplification in MoE Finalize + All-Reduce + Residual + RMSNorm** (`flashinfer-ai/flashinfer@7888b26bf6b5`, `#7298`): Across `H=7168` and `world_size ∈ {2, 4, 8}`, a pure one-shot schedule at `T ≥ 64` and `WS ∈ {4, 8}` forces every rank to pull `(WS - 1)` full token rows over NVLink.
3. **Register Pressure in Wide-Vocab (128K–256K) Cached-Cluster Softmax** (`flashinfer-ai/flashinfer@ce1758b8b38f`, `#7308`): Holding all 8 tiles (64 FP32 elements/thread) in registers caps occupancy at 2 CTAs/SM, while persistent cluster kernels require 67.7–68.7 KiB of dynamic shared memory per CTA.
4. **MiniMax-H3 Diffusion MLP (`H=5376, I=14336`) FC1 Output Tile Width & SM120 FP8 Varlen Attention PV Hand-Off** (`flashinfer-ai/flashinfer@fd0e4fdd96b0` `#7304`, `b8ed186df31b` `#7334`, `53ecbde224f4` `#7333`).

---

## 2. First-Hand Source Code & Data Layout Audit

### 2.1 NVFP4 MLA Decode: Native `E2M1 × E4M3` `mul.rn.bf16x2` Dequant & `cake-build/1` Contract (`#7306`)

Auditing `flashinfer/experimental/nvfp4_mla_decode/cake_backend.py`, `cake_jit.py`, and `csrc/cake_nvfp4_mla_decode/cake_nvfp4_mla_decode_4cd8124d834b7ab3a055_kernel.cu`:

- **Memory Layout & Byte Footprint**: Each 512-element NVFP4 latent token stores $512 \times 0.5\text{ B} = 256\text{ B}$ of packed `E2M1` payloads plus $512 / 16 = 32\text{ B}$ of `UE4M3` scale factors (**$288\text{ B/token}$** total) across 64-token pages (`PAGE_SIZE = 64`).
- **Native BF16x2 Dequantization**: Kernel `4cd8124d834b7ab3a055` converts `E2M1` nibbles and `UE4M3` block scales into paired `bf16x2` operands and executes native `mul.rn.bf16x2` instructions alongside a shortened `ex2.approx.ftz.f32` online softmax chain.
- **Self-Contained `cake-build/1` Loader**: Adds `cake_build.py` and `build.json` so prebuilt `aot/<name>.so` binaries validate `build_cuda_version()` against `aot/build_info.json` without importing `flashinfer` at runtime.
- **Author-Reported Benchmarks (`author-benchmark`, distinct from local CPU modeling)**:
  - **GB300 (`sm_103a`)**: **`1.046x–1.124x`** speedup across all 8 shapes (e.g., `B=32, q_len=1, kv_len=65536`: `598.32 -> 532.19 us`, **`1.124x`**, achieving **`3.5481 TB/s`** effective KV bandwidth; `B=32, kv_len=131072`: `1189.57 -> 1058.39 us`, **`1.124x`**).
  - **B200 (`sm_100a`)**: **`1.029x–1.074x`** speedup (`B=32, q_len=1, kv_len=8192`: `105.30 -> 98.02 us`, **`1.074x`**).

### 2.2 Latency-Ordered One-Shot & Two-Shot MoE All-Reduce Union (`#7298`)

Auditing `flashinfer/jit/cake_trtllm_moe_allreduce_union.py`:

- **Fused Operator Scope**: Combines routed-expert weighted reduction (`active_expert_tokens * expert_scales`), cross-rank all-reduce, residual addition, and RMSNorm in a single 224-thread, 4-CTA cluster launch (`896` threads/cluster).
- **One-Shot vs. Two-Shot Remote NVLink Read Formula**: For $T$ tokens, $H=7168$, and element size $b=2\text{ B}$:
  $$\text{Bytes}_{\text{1-shot}} = (W - 1) \cdot T \cdot H \cdot b, \qquad \text{Bytes}_{\text{2-shot}} = 2 \cdot \frac{W - 1}{W} \cdot T \cdot H \cdot b, \qquad \frac{\text{Bytes}_{\text{1-shot}}}{\text{Bytes}_{\text{2-shot}}} = \frac{W}{2}$$
  Two-shot cuts remote NVLink reads by **`2.0x`** at $W=4$ and **`4.0x`** at $W=8$.
- **Schedule Selection**: Uses latency-ordered one-shot (`lat_b4`, `lat_late_b5`) at small token counts and switches to two-shot (`two_shot_b4`, `two_shot_b5`) at `WS ∈ {4, 8}, T ≥ 64`.
- **Author-Reported Benchmarks (`author-benchmark`)**:
  - `WS=8, fp16, T=2048, E=8, PDL=off`: B200 drops from `357.76 us` to **`135.26 us`** (**`2.645x`**); B300 drops from `357.92 us` to **`134.14 us`** (**`2.668x`**).
  - `WS=8, bf16, T=256, E=8, PDL=off`: B200 drops from `60.35 us` to **`26.08 us`** (**`2.314x`**).

### 2.3 Shared-Tile Cluster Softmax (`#7308`) & MiniMax-H3 / SM120 Kernels (`#7304`, `#7334`, `#7333`)

- **Shared-Tile Cached-Cluster Softmax (`csrc/cake_blackwell_softmax.cu`)**:
  - Introduces `SoftmaxRoute::kCachedClusterSharedTiles = 7` (`c8_t4_s4` with `34,432 B` SMEM and `c16_t4_s4` with `34,944 B` SMEM).
  - Holds 4 tiles in registers and up to 4 tiles in shared memory via 1D `cp.async.bulk`, doubling occupancy from `2` to **`4 CTAs/SM`** and achieving **`1.173x`** on GB300 (`32 × 256000`: `15.20 -> 12.96 us`) and **`1.081x`** on B200 (`16 × 256000`: `10.69 -> 9.89 us`).
- **MiniMax-H3 `256 × 448` MXFP8 FC1 (`#7304`)**: Widens the 2-CTA output tile from `256 × 256` (`112` N-tiles) to **`256 × 448`** (**`64` N-tiles**) with staggered accumulator halves, reaching **`1.060x`** on B200 and **`1.068x`** on B300 (`rows=4096`: `208.16 -> 194.88 us`).
- **SM120 FP8 Varlen Attention (`#7334`)**: Overlap of next-unit Q/K0/V0 TMA prefetch and early PV hand-off yields **`1.055x`** on RTX 5090 (`34.91 -> 33.09 us`) and **`1.041x`** on RTX PRO 6000 (`27.68 -> 26.59 us`).

---

## 3. Reproducible CPU/Offline Verification

```powershell
python experiments\flashinfer_nvfp4_mla_decode_moe_allreduce_union_softmax_shared_tile_minimax_h3\run.py --check
```

| Subsystem / Metric | Configuration | Baseline | Optimized | Speedup / Ratio | Evidence Class |
|---|---|---:|---:|---:|---|
| NVFP4 MLA Decode (`#7306`) | GB300, `B=32, q=1, kv=64K` (`288 B/tok`) | `598.32 us` | `532.19 us` | **`1.1243x`** (`3.55 TB/s`) | `author-benchmark` + `own-offline` |
| NVFP4 MLA Decode (`#7306`) | B200, `B=32, q=1, kv=8K` | `105.30 us` | `98.02 us` | **`1.0743x`** | `author-benchmark` + `own-offline` |
| MoE AllReduce Union (`#7298`) | B300, `WS=8, fp16, T=2048, E=8` (`two_shot_b5`) | `357.92 us` | `134.14 us` | **`2.6683x`** (`4.0x` traffic cut) | `author-benchmark` + `own-offline` |
| Shared-Tile Softmax (`#7308`) | GB300, `32 × 256000`, `c16_t4_s4` (`34,944 B` SMEM) | `15.20 us` | `12.96 us` | **`1.1728x`** (`4 CTAs/SM`) | `author-benchmark` + `own-offline` |
| MiniMax-H3 MXFP8 FC1 (`#7304`) | B300, `rows=4096, 256×448 tile` (`64` N-tiles) | `208.16 us` | `194.88 us` | **`1.0681x`** | `author-benchmark` + `own-offline` |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Sizing Exercise**: For `world_size = 8`, `H = 7168`, `bf16`, and `T = 512`, compute per-rank remote NVLink read bytes under One-Shot vs. Two-Shot.
   - **Solution**: Local row payload is $512 \times 7168 \times 2 = 7.0\text{ MiB}$. One-Shot reads $7 \times 7.0 = \mathbf{49.0\text{ MiB}}$; Two-Shot reads $2 \times \frac{7}{8} \times 7.0 = \mathbf{12.25\text{ MiB}}$ (**`4.0x`** reduction).
2. **Falsifiable Hypothesis**: Does `cached_shared_tiles_kernel()` activate when `variant.max_tiles == 4`?
   - **Verification**: False. `if (variant.max_tiles != kCachedMaxTiles) return nullptr;` (`kCachedMaxTiles = 8`) restricts shared-tile routing to 8-tile chunks because 4-tile chunks already sustain 4 CTAs/SM in registers.
