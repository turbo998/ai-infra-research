# Experiment: `mi300x_large_moe_aipc_kev_sft`

Deterministic verification of:
1. **AIPC Edge Decision Model (`Kev-0.8B` / `Kev-4B`)**: Counterfactual SFT accuracy progression (`+51.6 pts` and `+31.5 pts`), post-hoc temperature scaling ($T = 2.83$, `-58.0%` ECE reduction), and 8-bit `Q8_0` GGUF CPU RAM footprint (`2.15 GiB` on Snapdragon X Elite within the `2–3 GB` budget).
2. **Azure AMD MI300X (`Standard_ND96isr_MI300X_v5`) Large MoE Validation**: BF16 vs. OCP MXFP4 weight footprint (`-68.3%` to `-71.5%`), single-node `TP=8` vs. dual-node `TP=8, PP=2` capacity boundaries, and throughput/quality validation across `Muse Spark Open 816B-A42B`, `Nemotron 3 Ultra 550B-A55B`, and `Inkling 975B-A41B`.

## Usage

```bash
python experiments/mi300x_large_moe_aipc_kev_sft/run.py --check
```
