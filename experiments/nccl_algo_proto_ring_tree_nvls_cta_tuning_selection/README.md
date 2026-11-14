# NCCL Collective Algorithm (`Ring` vs. `Tree` vs. `NVLS` vs. `PAT`), Protocol (`LL` / `LL128` / `Simple`) & Zero-SM CTA Tuning Selection (`nccl_algo_proto_ring_tree_nvls_cta_tuning_selection`)

Deterministic Python 3.12 analytical evaluator for **Round 177**: comparing `Naive 32-Channel Ring + Simple` (Tier 1), `Double Binary Tree + Auto LL/LL128/Simple` (Tier 2), `NVLS + PAT Binomial Tree` (Tier 3), and `Zero-Copy User Buffer Registration (ncclCommRegister) + NVLS + Copy Engine P2P DMA` (Tier 4).

## Reproduction

```bash
python -m experiments.nccl_algo_proto_ring_tree_nvls_cta_tuning_selection.run
python -m unittest tests/test_nccl_algo_proto_ring_tree_nvls_cta_tuning_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_nccl_benchmarks`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`
