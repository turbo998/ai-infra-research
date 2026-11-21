# Diffusion LLM (`LLaDA / Mercury Coder / Block Diffusion`) vs. Autoregressive LLM Selection Experiment

Reproducible analytical model comparing 4 generation paradigms for `512-token` code/JSON generation with a `4,096-token` prompt on `NVIDIA H100 80GB`:
- **Tier 1 (`Standard Serial Autoregressive Decode`, 1 token/step)**: `110.0 TPS` per user (`4,654 ms` for 512 tokens), `62.5%` bidirectional infilling/reversal accuracy.
- **Tier 2 (`Autoregressive + EAGLE-3 / MTP Speculative Decoding`)**: `285.0 TPS` per user (`1,796 ms`).
- **Tier 3 (`Vanilla Full-Sequence Masked Diffusion LLM LLaDA` without KV Cache)**: `640.0 TPS` (`800 ms`), `94.2%` bidirectional infilling accuracy, but recomputes the prompt every step.
- **Tier 4 (`Semi-Autoregressive Block Diffusion BD3-LM / Mercury Style` + `Causal Prefix KV Cache` + `Confidence-Adaptive Early Unmasking`, Recommended)**: **`980.0 TPS` per user (`522 ms` for 512 tokens — `8.9x` faster than serial AR)**, full Prefix KV Cache reuse, and `93.8%` bidirectional infilling accuracy.

## Run & Verify

```bash
python experiments/diffusion_llm_llada_mercury_block_diffusion_vs_autoregressive_selection/run.py
python experiments/diffusion_llm_llada_mercury_block_diffusion_vs_autoregressive_selection/run.py --check
```
