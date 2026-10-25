# Architectural Audit: FlashInfer SM120 QSA NVFP4 Whole-Tile Decode, SM100/103 DCP All-to-All LL128 Deferred Release & W4A16 Split MoE

> **中文版**: [flashinfer-qsa-nvfp4-sm120-wholetile-dcp-a2a-ll128-w4a16-split-moe.md](flashinfer-qsa-nvfp4-sm120-wholetile-dcp-a2a-ll128-w4a16-split-moe.md)

## 1. Context & Architectural Bottlenecks

Across Blackwell workstation/server GPUs (SM120, e.g., NVIDIA RTX PRO 6000 Blackwell Server Edition with 188 SMs) and datacenter B200/GB300 GPUs (SM100/SM103), three low-level execution boundaries govern tail latency in sparse paged attention (QSA), decode context parallelism (DCP), and W4A16 Mixture-of-Experts (MoE):
1. **SM120 QSA NVFP4 Long-Stream L2 Thrashing**: When decoding 2051 selected sparse tokens per row (33 tiles of 64 tokens) with `Hq=24, Hkv=2, D=256`, the fused cooperative K-first path suffers L2 cache eviction pressure at wider batch sizes (`rows >= 32`) where each CTA walks a long stream of tiles.
2. **SM100/SM103 DCP All-to-All LL128 FIFO Tail-Slot Synchronization**: In `decode_cp_a2a_alltoall` (`cake_dcp_alltoall`), exchanging BF16 partial outputs (`D=128`) and two FP32 softmax statistics across CP2/CP4 NVLink ranks previously required releasing the final LL128 FIFO slot before kernel exit.
3. **W4A16 Split MoE Crossover vs. SM103 NVFP4 Attention TMEM Allocation**: Introducing a two-kernel BF16 × NVFP4 "W4A16 Split" MoE backend (`cf5c9a6a41`) and expanding SM103 TMEM columns from 432 to 480 to hold `P` in tensor memory (`d08a1f7321`).

---

## 2. First-Hand Source Code Audit & Hardware Mechanics

### 2.1 SM120 QSA NVFP4 Whole-Tile Decode (`flashinfer-ai/flashinfer` @ `83755d9093`)
Auditing `flashinfer/experimental/cake_qsa_nvfp4/cake_backend.py` and `README.md`:
- **Co-Resident CTA Invariant**: On SM90 and SM120, `FUSED_CTAS_PER_SM = 2` (yielding `188 × 2 = 376` co-resident CTAs on RTX PRO 6000, compared to `148` on B200, `152` on GB300, and `212` on R200).
- **Whole-Tile Prefetch & L2 Evict-First**: The new SM120 `decode_whole_tile` program (shipped in split form: decode + merge) issues every copy of a 64-token tile before that tile's compute, keeps two pipeline stages in flight, and applies an L2 `evict-first` hint to the copies.
- **11-Tile (`704` Selected Tokens) Selection Rule**: The host selects `decode_whole_tile` on SM120 whenever `ceil(tiles / splits) >= 11` (`704` tokens), which covers `rows = 32 / 64 / 128` at `3 / 2 / 1` splits:
  - Eager launches accelerate by **1.05x–1.06x** (`161.65 -> 152.80 µs` at NHD `rows=128`, `168.11 -> 158.50 µs` at HND `rows=128`);
  - CUDA-graph replay accelerates by **1.05x–1.07x** (`100.35 -> 93.86 µs` at NHD `rows=64`), while `rows <= 16` stay on the fused cooperative K-first path (`0.99x–1.01x`).

### 2.2 SM100/SM103 DCP All-to-All LL128 Deferred Slot Release (`flashinfer-ai/flashinfer` @ `53feaae34c`)
Auditing `csrc/cake_dcp_alltoall_dispatch.cu` and `cake_dcp_alltoall_ff3dfded867e5738be23_kernel.cu` (CP2) / `b8b8ebc8b91fd813f375` (CP4):
- **Next-Launch Slot Release & Direct Register Stores**: The receiver releases its last LL128 FIFO slot at the **beginning of the next launch** rather than stalling at the tail of the current launch, and writes received entries directly from registers via plain stores.
- **Measured Gains on GB300 (`sm_103a`) and GB200 (`sm_100a`)**:
  - GB200 CP2 `tokens=1`: Warm L2 drops from `5.456 µs` to `4.288 µs` (**1.27x**); Cold L2 drops from `5.584 µs` to `4.192 µs` (**1.33x**);
  - GB300 CP4 `tokens=1..128`: **1.16x–1.21x** faster across all tested shapes.

### 2.3 Cake W4A16 Split MoE & SM103 NVFP4 Attention P-in-TMEM (`cf5c9a6a41`, `d08a1f7321`)
- **W4A16 Split MoE (`cf5c9a6a41`)**: On B300 (`sm_103a`), Cake's two-kernel W4A16 Split MoE beats FlashInfer's CuTe DSL W4A16 baseline at low token counts (`D1`: `53.73 -> 50.98 µs`, **1.054x**; `G8` EP8: `58.63 -> 55.98 µs`, **1.047x**) and very large batches (`G4096`: **1.023x**), but trails at `G128` (`123.31 vs 133.62 µs`, **0.923x**).
- **SM103 NVFP4 Attention (`d08a1f7321`)**: Expanding `TMEM_NCOLS` from `432` to `480` (`TMEM_TMEM_P0_LO/HI_OFFSET = 448..472`) to hold `P` in TMEM yields **1.02x–1.05x** speedup on GB300 (`0.1145 -> 0.1094 ms` at `(4, 8, 4096)`).

---

## 3. Deterministic Verification (`flashinfer_qsa_nvfp4_sm120_wholetile_dcp_a2a_ll128_w4a16_split_moe`)

| Subsystem / Metric | Baseline Path | Phase 81 Optimized Path | Verified Result | Source ID |
|---|---|---|---|---|
| SM120 QSA NVFP4 (`rows=64`, Graph) | K-First Split (`100.35 µs`) | Whole-Tile + L2 Evict-First (`>=11` tiles/CTA) | **93.86 µs (1.07x)** | `src-flashinfer-cake-qsa-nvfp4-sm120-wholetile-2026` |
| GB200 DCP All-to-All (CP2, `T=1`) | Tail-Sync Slot Release (`5.584 µs` Cold) | Next-Launch Slot Release + Reg Stores | **4.192 µs (1.33x)** | `src-flashinfer-cake-dcp-alltoall-sm100-sm103-2026` |
| B300 W4A16 Split MoE (`D1` / `G8`) | CuTe DSL W4A16 (`53.73 / 58.63 µs`) | Cake W4A16 Split (`50.98 / 55.98 µs`) | **1.054x / 1.047x** (`G128`: `0.923x`) | `src-flashinfer-cake-fused-moe-w4a16-split-2026` |
| GB300 NVFP4 Attention (`4×8×4096`) | `TMEM_NCOLS=432` (`0.1145 ms`) | `TMEM_NCOLS=480` + P-in-TMEM (`0.1094 ms`) | **1.05x** | `src-flashinfer-cake-nvfp4-attn-sm103-tmem-p-2026` |
