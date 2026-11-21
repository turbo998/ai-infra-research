# Sub-Quadratic Hybrid SSM & Linear Attention Selection Experiment (`Mamba-2 / Jamba / Qwen3-Next Gated DeltaNet / Lightning Attention` vs. Pure `GQA` Transformer)

Reproducible HBM capacity, `256K` Prefill `TTFT`, batch concurrency, and `RULER` needle-in-a-haystack accuracy model comparing 4 sequence-modeling architectures across `256K`-token enterprise contexts.

## Run
```bash
python experiments/sub_quadratic_hybrid_ssm_mamba2_jamba_qwen3_next_linear_attention_selection/run.py
python experiments/sub_quadratic_hybrid_ssm_mamba2_jamba_qwen3_next_linear_attention_selection/run.py --check
```
