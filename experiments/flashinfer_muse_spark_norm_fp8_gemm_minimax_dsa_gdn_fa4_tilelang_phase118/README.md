# Phase 118 Track A Experiment Suite

Deterministic Python stdlib simulation covering 10 upstream kernel commits (`priority: 127`):
- FlashInfer `cake_muse_spark_norm` (`b08e86326a`, #7505)
- FlashInfer `cake_grouped_fp8_gemm` (`ccb6fc8d4c`, #7597)
- FlashInfer `cake_minimax_h3` (`574d6d8d53`, #7599)
- FlashInfer `cake_dsa_indexer` (`b532fa4443`, #7571)
- FlashInfer `cake_moe_finalize_allreduce_fusion` (`3cf451835d`, #7511)
- FlashInfer `cake_gdn_det_train` (`6b6c9f996d`, #7592)
- FlashInfer `cake_nvfp4_mla_decode` (`ee76d93ae8`, #7595)
- FlashInfer `moe` W4A16 scale reduction (`94090279f3`, #4559)
- FlashAttention-4 SM100 Sparse MLA backward (`ce1d8313d5`)
- TileLang SIMT atomic region shape validation (`af08dae6e4`, #3465)

## Usage

```bash
python experiments/flashinfer_muse_spark_norm_fp8_gemm_minimax_dsa_gdn_fa4_tilelang_phase118/run.py --write
python experiments/flashinfer_muse_spark_norm_fp8_gemm_minimax_dsa_gdn_fa4_tilelang_phase118/run.py --check
```
