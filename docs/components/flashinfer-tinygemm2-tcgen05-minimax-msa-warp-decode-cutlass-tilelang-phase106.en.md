# FlashInfer `cake_tinygemm2_tcgen05` Swap-AB Tile, SM120 Proxy Fence, SM90 MSA, `cake_warp_decode` Finalize Spread & CUTLASS/TileLang Analysis (Phase 106 · Priority 115)

> **中文版**: [flashinfer-tinygemm2-tcgen05-minimax-msa-warp-decode-cutlass-tilelang-phase106.md](flashinfer-tinygemm2-tcgen05-minimax-msa-warp-decode-cutlass-tilelang-phase106.md)  
> **Evidence Boundary**: `implementation` (first-hand upstream commits) + `own-offline` (deterministic standard-library model: `experiments/flashinfer_tinygemm2_tcgen05_minimax_msa_warp_decode_phase106/run.py --check`)

## 1. Primary Source Evidence & Implementation Mechanics

1. **`flashinfer-ai/flashinfer@503b948ac73d7321622c750b5570ddd9ed6e3a5e` (`csrc/cake_tinygemm2_tcgen05/*`)**:
   - **Swap-AB Accumulator Tile, 32-Byte Vector Store & Early TMEM Free**: Swaps MMA operands on the 64-row per-chain-accumulator tile (`M=64` activations as operand `A`, `N=32` weights as operand `B`) so each lane writes its 16 BF16 outputs as a single **32-byte vector store (`16 -> 1` store instruction, `16.0x` store instruction reduction)** when `out.data_ptr() % 32 == 0`, while freeing Tensor Memory right after the epilogue's last TMEM read (`28.0 -> 19.6 us`, **`1.4286x` speedup**).
2. **`flashinfer-ai/flashinfer@8a087fcc87b1e7baed17c67603fe3051c5066d8a` & `47c17dd7aef8d9084e645c3f2e6ea3d32682b084` (`csrc/cake_minimax_h3_sm120_*`)**:
   - **SM120 Ring-Stage Proxy Fence & NVFP4 Pre-Attention Tile Order**: Inserts `fence.proxy.async.shared::cta` after each `ab_empty` wait before TMA refills the ring stage in SM120 FP8/NVFP4 MLP kernels (`18 -> 0` ring-stage hazards across `10,000` iterations) and reorders SM120 NVFP4 pre-attention GEMM raster tiles for a **`1.0494x`** speedup on RTX 5090.
3. **`flashinfer-ai/flashinfer@b3059f79e1fd77541f6cdddb78672086eca405c1` (`flashinfer/msa_ops/_sm90_dispatch.py`, `cute_dsl/*_sm90.py`)**:
   - **Hopper SM90 MiniMax-M3 Sparse Attention (MSA) CuTe DSL Kernels**: Implements native SM90 CuTe DSL backends (`proxy_score`, `topk=16`, `sparse_prefill`, `sparse_decode`) over interleaved paged FP8 E4M3 KV caches, replacing the H100 Triton fallback (`112.0 -> 44.8 us`, **`2.50x` speedup**).
4. **`flashinfer-ai/flashinfer@ed8b6b956dbcba4729a1378585e49902c9308761` & `2725b4ec0e378700c415e3354bdc90307f8bca4f` (`csrc/fused_moe/warp_decode/*`)**:
   - **11-Tier Dynamic SMEM Finalize Grid Spreading & Stream-Order Chain Head Launch**: Reserves dynamic shared memory in 11 tiers (`118,784 B` down to `19,456 B`) to cap per-SM occupancy at `1..11` CTAs and spread 64 finalize CTAs across **`64` SMs instead of `8` SMs (`8.0x` SM spread, `18.0 -> 6.0 us` `3.0x` finalize speedup)**, while launching read-first chain heads in stream order rather than PDL (`12 -> 0` stale `topk_ids` races).
5. **`NVIDIA/cutlass@ffa119a125`, `7107b05535` & `tile-ai/tilelang@03211b85cc`, `eaa9468d60`**:
   - Enables `reassoc` fastmath on `CuTeDSL` `Vector.reduce()` (`8 -> 3` tree reduction steps, **`2.6667x`** depth reduction), matches `int64_t` C header shape slots for `cute.sym_int64()` dynamic dimensions (`0` struct offset misreads), hashes ordered `compile_flags` in TileLang's autotuner cache identity (`0` binary collisions), and detects same-iteration cross-thread shared-memory WAW hazards (`100%` synchronization coverage).
