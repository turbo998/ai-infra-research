# FlashInfer SM103 `mhc_fused_post_pre`, Cooperative `kimi_k3_tp12_tail`, 10/12-CTA Softmax, SM103a `K=96` NVFP4 SVDQuant MMA & TileLang Negative-Index Guard (Phase 105)

Deterministic standard-library verification for Phase 105 (`priority: 114`) Track A:
- **FlashInfer SM103 `mhc_fused_post_pre`** (`f56fecc7ebf7989a5f075031ff0ded26000b1e8b`): Fuses 3 kernels (`mhc_post`, TF32 pre-norm GEMM, `mhc_pre_big_fuse_with_prenorm`) across `89` boundaries (`267 -> 89` launches, `-178` launches/pass), cutting `T=11,476` HBM traffic from `1.69 -> 0.94 GB` (`-44.38%`, `1.7979x`) and accelerating GB300 decode (`T=48`) by `2.1628x` (`14.88 -> 6.88 us`) and mixed (`T=11,476`) by `1.6694x` (`310.96 -> 186.27 us`).
- **FlashInfer Cooperative `kimi_k3_tp12_tail` (`9a27746f1d`), 10/12-CTA Softmax (`e61cdfa925`) & `K=96` SVDQuant (`4f3b2d6b91`)**: Eliminates undersized green-context spin-wait deadlocks (`1 -> 0`), accelerates B200 cached-cluster softmax by `1.1692x` (`9.95 -> 8.51 us`), and speeds up SM103a NVFP4 SVDQuant by `1.0955x` (`162.46 -> 148.30 us`) with `96+96+64` (`4 -> 3` MMAs per 256-K stage).
- **CUTLASS (`147295a3d4`, `59e3a3338d`, `b87a77760f`) & TileLang (`004b7a2a69`, `fa456b8b2c`)**: Restores CuTeDSL `opt_level=2` on CUTLASS `>= 4.7` and rejects out-of-range negative constant indices (`index + extent < 0`).

```bash
python experiments/flashinfer_mhc_post_pre_tp12_tail_softmax_svdquant_phase105/run.py --check
```
