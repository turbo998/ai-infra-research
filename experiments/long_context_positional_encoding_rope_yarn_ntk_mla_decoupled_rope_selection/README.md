# Long-Context Positional Encoding & KV Scaling Experiment (`RoPE` vs. `Dynamic NTK` vs. `YaRN` vs. `MLA Decoupled RoPE`)

Reproducible analytical model comparing 4 positional encoding architectures when extending context `16x` (`8K -> 128K`):
- **Tier 1 (`Unscaled RoPE / Linear Position Interpolation PI`)**: `14.5%` 128K NIAH recall, `86.2%` short-context retention.
- **Tier 2 (`Dynamic NTK-Aware Scaling`)**: `78.4%` 128K NIAH recall, `97.5%` short-context retention.
- **Tier 3 (`YaRN / Llama-3 Piecewise RoPE + Attention Logit Temperature mscale`)**: `98.2%` 128K NIAH recall, `99.4%` short-context retention (`2,048` KV elements/token/layer for GQA-8).
- **Tier 4 (`MLA Decoupled 64-Dim Shared RoPE + 512-Dim NoPE Latent c_KV + YaRN`, Recommended)**: `99.1%` 128K NIAH recall, `99.6%` short-context retention, enables `W_UK` matrix absorption into `W_Q`, and shrinks KV Cache to `576` elements/token/layer (`3.55x` smaller than GQA-8).

## Run & Verify

```bash
python experiments/long_context_positional_encoding_rope_yarn_ntk_mla_decoupled_rope_selection/run.py
python experiments/long_context_positional_encoding_rope_yarn_ntk_mla_decoupled_rope_selection/run.py --check
```
