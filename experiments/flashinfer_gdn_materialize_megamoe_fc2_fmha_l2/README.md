# FlashInfer GDN Prefix-Cache State Materialization, Cake W8A8 MegaMoE EP16 FC2 Evict-First, and Cake FMHA Uniform L2 Stream Hints

Deterministic Python 3.12 standard-library model verifying:
1. **`flashinfer-ai/flashinfer` `99406bb9cae63a7154dc3498460f4778f1b8a33b` (`#5073`)**: `gdn_prefix_materialize` partial ring fold $S_C = \exp(G_C) S_0 + \sum_{j<C} \exp(G_C - g_j) u_j \otimes k_j$ at prefix-cache block boundaries and persistent-grid `active=0` launch overhead reduction ($32.0\,\mu\text{s} \to 2.5\,\mu\text{s}$).
2. **`flashinfer-ai/flashinfer` `60213537ce15b522fa75e4526187f505ec36ff70` (`#6857`)**: `cake_w8a8_megamoe_ep16` FC2 4D TMA `L2::cache_hint` (`0x12F0000000000000ULL`, `evict_first`) on expert weights `A` and scales `SFA`.
3. **`flashinfer-ai/flashinfer` `566812deed79eff36b6c1298be9e9ea2d4f101e9` (`#6111`)**: `cake_fmha` balanced decode `plan_flags` (`work_token_words[base + 11]`) uniform-gated L2 `evict_first` and V-page prefetch hints.

## Run
```bash
python experiments/flashinfer_gdn_materialize_megamoe_fc2_fmha_l2/run.py
python experiments/flashinfer_gdn_materialize_megamoe_fc2_fmha_l2/run.py --check
```
