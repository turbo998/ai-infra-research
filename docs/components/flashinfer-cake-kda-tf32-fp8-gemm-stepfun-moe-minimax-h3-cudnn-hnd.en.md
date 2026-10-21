# FlashInfer Cake KDA TF32 24->9 Kernel Folding, 4-Family FP8 GEMM 56-Row Routing Table, StepFun MoE `cake-build/1` 222-Entry Manifest, MiniMax H3 QKV 8-Variant Pack & cuDNN Paged Prefill HND Layout Contract

> **Phase 55 (`priority: 64`) · First-Hand Commit-Level Deep Dive**
> - `flashinfer-ai/flashinfer@9f7a290987b69277a9031b1c6091810d83cfed3b` (`#6876`)
> - `flashinfer-ai/flashinfer@272ed2a6cf2fe66356b31b91635c26220b819fc8` (`#6862`)
> - `flashinfer-ai/flashinfer@e1b052de722e193324559ba2639b26992c7b9f50` (`#6871`)
> - `flashinfer-ai/flashinfer@5ea501832bf7e37a7acde903a32329d580c0e9b4` (`#6869`)
> - `flashinfer-ai/flashinfer@013f02e75ddf8430ad010b965b2d5047c370f576` (`#6846`)

---

## 1. Cake KDA TF32 Prefill Kernel Folding (`#6876`)

In the Kimi Delta Attention (KDA) TF32 prefill family, `#6876` applies three structural mechanics:
1. **Shared Device Header (`csrc/kda/tf32/cake_kda_device.cuh`, `1,039` lines)**: Hoists inline PTX helpers (`elect_sync()`, `mbarrier_init()`) out of all 34 TF32 kernel sources into a single header.
2. **Folding 24 Clones onto 9 Shared Bodies**: Parameterizes identical kernel bodies via compile definitions (`CAKE_KDA_KERNEL_SYMBOL`, `CAKE_KDA_WRITE_CHECKPOINTS`), shrinking `csrc/kda/tf32/` from **35 entries / 97,009 lines** to **21 entries / 41,231 lines** (**-57.50%**) while rendering all 34 TF32 host bindings from one template (`-9,497` lines).
3. **Dead Runtime Pruning**: Removes `prepare_fwd`, `KDAInputABI`, `_rebind_tensor_signature`, 15 probe launch classes, and 19 unused constants, trimming the shared runtime from `6,478` to `5,832` lines.

---

## 2. Four-Family Cake FP8 GEMM Routing & StepFun SwiGLU Fused-MoE (`#6862`, `#6871`)

- **Four-Family Cake FP8 Consolidation (`#6862`)**: Covers `batch_deepgemm_fp8` (`38` programs, `67` modules across `sm_100a`/`sm_103a`), `cake_grouped_fp8_gemm` (`15` programs, folding 3 `deepk_c2` variants onto 1 source), `cake_grouped_fp8_fused_silu_quant` (`5` programs), and `cake_grouped_mxfp8_quantize` (`2` programs, building FP16 from the BF16 source) — **60 programs** total. Replaces two rendered routing chains with a **56-row lookup table**, shrinking the batch loader from `6,381` to `1,548` lines (**-75.74%**).
- **StepFun SwiGLU Fused-MoE Round-2 (`#6871`)**: Serves `H=4096, I=1536, E=64, top_k=8` across `bf16 / fp8 / mxfp8 / nvfp4` via **111 programs** (`routing 52 + FC1 26 + FC2 27 + requant 2 + finalize 4`) and **222 content-sealed `build.json` manifest entries** (`sm_100a` + `sm_103a`), sharing `47` duplicate host bindings and shrinking `cake_jit.py` (`8,064 -> 1,212` lines) and `cake_launch.py` (`5,340 -> 21` lines).

---

## 3. MiniMax H3 Fused QKV Quantize+Pack & cuDNN Paged Prefill HND Contract (`#6869`, `#6846`)

- **MiniMax H3 `minimax_h3_qkv_quantize_pack` (`#6869`)**: Ships all **8 variants** (`{MXFP8, NVFP4} × P ∈ {1, 2, 4, 8}`) on `sm_100a` and `sm_103a` through the in-tree program loader with `1.000x` geomean parity across `M = 33,472..109,952`.
- **cuDNN Paged Prefill HND Layout (`#6846`)**: Declares `"HND"` instead of `"NHD"` for `[pages, heads=8, page_size=16, dim]` tensors, preventing axis-swap corruption (`kv_heads=16, page_size=8`) that broke GQA divisibility for `qo_heads=40` (`40 % 16 != 0` vs valid `40 / 8 = 5`).

---

## 4. Deterministic Verification

Run `python experiments/flashinfer-cake-kda-tf32-fp8-gemm-stepfun-moe-minimax-h3-cudnn-hnd/run.py --check` to reproduce all verified metrics in `summary.json`.
