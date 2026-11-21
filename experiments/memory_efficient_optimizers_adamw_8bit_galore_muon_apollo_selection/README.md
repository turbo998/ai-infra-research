# Memory-Efficient & Orthogonalized Optimizers Selection Experiment (`FP32 AdamW` vs. `8-Bit AdamW` vs. `GaLore/APOLLO` vs. `Distributed Muon`)

Reproducible analytical model comparing 4 LLM training optimizer architectures on a `70B` model:
- **Tier 1 (`Standard FP32 AdamW` `12Ψ` state)**: `840.0 GB` optimizer state, `1.00x` token efficiency.
- **Tier 2 (`Blockwise 8-Bit Quantized AdamW` `6Ψ` state)**: `420.0 GB` optimizer state (`-50%` HBM & checkpoint size), `1.00x` token efficiency.
- **Tier 3 (`GaLore / APOLLO Low-Rank Gradient Projection` `2.4Ψ` state)**: `168.0 GB` optimizer state (`-80%` HBM), enables full-parameter updates on smaller clusters.
- **Tier 4 (`Distributed Muon`: `5-Step BF16 Newton-Schulz Orthogonalized Momentum` on 2D layers without $v_t$ second moment + `AdamW` on 1D/Embeddings, Recommended)**: `560.0 GB` optimizer state (`8Ψ`, `-33.3%` HBM vs. AdamW) and **`1.42x` higher token sample efficiency (reaches target validation loss with `29.5%` fewer total GPU hours!)**.

## Run & Verify

```bash
python experiments/memory_efficient_optimizers_adamw_8bit_galore_muon_apollo_selection/run.py
python experiments/memory_efficient_optimizers_adamw_8bit_galore_muon_apollo_selection/run.py --check
```
