# Enterprise LLM Quantization Selection (`BF16` vs. `W4A16 AWQ/Marlin` vs. `Block-FP8 W8A8` vs. `Blackwell NVFP4`) & Batch-32 Crossover Model (`#457`)

Reproducible analytical model comparing unquantized `BF16`, weight-only `W4A16 AWQ/Marlin`, fine-grained `Block-FP8 (1x128/128x128)`, and hardware-native `NVFP4 / MXFP4 (W4A4)` across low-batch (`B=8`) and high-concurrency (`B=64`) regimes.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_quantization_bf16_w4a16_block_fp8_nvfp4_batch_crossover_sizing_selection/run.py --check`
