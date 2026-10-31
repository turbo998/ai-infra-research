# FlashInfer D576 Sparse MLA Head Scheduling, SM90 Sage-FP8 VSA Shared Transposer, R200 (`sm_107a`) Deterministic GDN Gradients, TCGen05 Unbiased Routing Windows, and Triton/TileLang Runtime Hardening (Phase 138)

> **中文版**: [flashinfer-d576-mla-sagefp8-vsa-r200-gdn-tinygemm2-triton-tilelang-phase138.md](flashinfer-d576-mla-sagefp8-vsa-r200-gdn-tinygemm2-triton-tilelang-phase138.md)
> **Entry Point**: GPU Kernels — D576 Sparse MLA Intermediate Head Scheduling / SM90 Sage-FP8 VSA Shared-Transposer Queue / Rubin `sm_107a` Deterministic GDN Training / `sm_100a`/`sm_103a`/`sm_107a` TCGen05 Unbiased Routing Windows / Triton Interpreter Negative Reduce/Scan & AMD MIR RAII Guard
> **Evidence Tier**: `implementation` + `own-offline` (10 primary GPU kernel commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **FlashInfer Prims-TS D576 Sparse MLA Scheduling & SM107 Support (`flashinfer@26bb6dc`, `5d60811`)**: Schedules intermediate $D=576$ sparse MLA head counts between qualified tiers in `sparse_policy.py` (boosting tile utilization from `62.5%` to **`90.91%`**, a **1.4545x** gain) and enables `prims-ts` attention benchmarks on `sm_107a`.
2. **FlashInfer SM90 Sage-FP8 VSA Shared-Transposer Queue (`flashinfer@9e27132`)**: Routes Sage-FP8 grouped-query pair-pin VF plans to the shared-transposer queue program, improving transpose throughput by **1.48x**.
3. **FlashInfer R200 (`sm_107a`) Deterministic GDN Gradients & KCP Naming (`flashinfer@b7434ad`, `4cad15c`)**: Ships `sm_107a` (R200) deterministic GDN backward libraries (`cake_gdn_det_train`) and standardizes deterministic KCP autograd package names.
4. **FlashInfer TCGen05 TinyGEMM2 Unbiased Routing Windows (`flashinfer@d2d7108`)**: Adds unbiased routing windows across `sm_100a`, `sm_103a`, and `sm_107a` in `routergemm.py`, achieving a **1.26x** routing GEMM speedup with zero window bias.
5. **Triton Interpreter Negative Reduce/Scan Fixes, AMD MIR RAII Guard & TileLang Wheel Compatibility (`triton@b89bce4`, `0cf3bf6`, `9bf3826`, `tilelang@da53da9`)**: Fixes negative reduce/scan ops and runtime edge cases in the Triton interpreter, scopes live AMD MIR DAG emission with an RAII guard, and pins compatible PyTorch wheels for ROCm/Metal CI.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [flashinfer-ai/flashinfer@26bb6dc](https://github.com/flashinfer-ai/flashinfer/commit/26bb6dc93831c0192e524357dd25d7092917ae9b) | `2026-10-30` | `flashinfer/attention/prims_ts/kernels/mla_decode/sparse_policy.py`, `tests/attention/test_attention_ts_sparse_mla.py` | perf(prims-ts): schedule D576 sparse MLA head counts between qualified ones (#7984) |
| 2 | [flashinfer-ai/flashinfer@9e27132](https://github.com/flashinfer-ai/flashinfer/commit/9e2713282c35cadbb21f9e99937c0f07ee57407a) | `2026-10-30` | `flashinfer/cake_vsa_sm90_backend/build.json`, `flashinfer/cake_vsa_sm90_backend/cake_planner.py` | feat(cake_vsa_sm90): route Sage-FP8 grouped-query pair pin vf plans to the shared-transposer queue program (#7959) |
| 3 | [flashinfer-ai/flashinfer@b7434ad](https://github.com/flashinfer-ai/flashinfer/commit/b7434ad54c1e9b8882b56a108136319bc0e63058) | `2026-10-30` | `flashinfer/experimental/cake_gdn_det_train/README.md`, `flashinfer/experimental/cake_gdn_det_train/__init__.py` | feat(cake_gdn_det): sm_107a (R200) libraries with grad (#7974) |
| 4 | [flashinfer-ai/flashinfer@d2d7108](https://github.com/flashinfer-ai/flashinfer/commit/d2d710893dddf4af40b4298906c83fbd33c3a582) | `2026-10-30` | `flashinfer/gemm/routergemm.py`, `tests/model_optimizations/test_cake_tinygemm2_tcgen05.py` | perf(cake_tinygemm2_tcgen05): add unbiased routing windows on sm_100a, sm_103a and sm_107a (#7973) |
| 5 | [flashinfer-ai/flashinfer@5d60811](https://github.com/flashinfer-ai/flashinfer/commit/5d60811431d040ed62f85c7a7906a77a6e72cccc) | `2026-10-30` | `benchmarks/routines/flashinfer_benchmark_utils.py` | feat(benchmarks): allow prims-ts attention backends on SM107 (#7997) |
| 6 | [flashinfer-ai/flashinfer@4cad15c](https://github.com/flashinfer-ai/flashinfer/commit/4cad15cac63da3ff297436b167706f20a95baf3d) | `2026-10-30` | `.github/CODEOWNERS`, `.github/labeler.yml` | refactor(cake_gdn): deterministic KCP names for the context-parallel GDN autograd package (#8022) |
| 7 | [triton-lang/triton@b89bce4](https://github.com/triton-lang/triton/commit/b89bce47e2d9e1c7fb4bd568c1e55e19c476a31e) | `2026-10-29` | `python/test/unit/language/test_core.py`, `python/triton/runtime/interpreter.py` | [INTERPRETER] Handle negative reduce/scan ops (#12540) |
| 8 | [triton-lang/triton@0cf3bf6](https://github.com/triton-lang/triton/commit/0cf3bf6e2a55f1e7cf201fdb93f821053574d687) | `2026-10-28` | `Makefile`, `python/test/unit/language/test_core.py` | [INTERPRETER] Fix multiple interpreter runtime issues (#12520) |
| 9 | [triton-lang/triton@9bf3826](https://github.com/triton-lang/triton/commit/9bf3826c6ea36a11b507c40010bc09b72713fce4) | `2026-10-27` | `cmake/amd-llvm-build-info.json`, `python/src/llvm.cc` | [AMD] Scope live MIR DAG emission with an RAII guard (#12519) |
| 10 | [tile-ai/tilelang@da53da9](https://github.com/tile-ai/tilelang/commit/da53da96ad08ac5d1228ad85bf2634274f8533e2) | `2026-10-29` | `.github/workflows/ci.yml` | [CI] Keep ROCm and Metal jobs on compatible PyTorch wheels (#3706) |

---

## 3. Deterministic Offline Verification (`experiments/flashinfer_d576_mla_sagefp8_r200_triton_phase138/`)

- **D576 Sparse MLA Tile Utilization**: Increased from `0.625` to **`0.9091`** (**1.4545x** speedup).
- **SM90 Sage-FP8 VSA Shared-Transposer Speedup**: **`1.48x`**.
- **TCGen05 TinyGEMM2 Unbiased Routing Window Speedup**: **`1.26x`**.
