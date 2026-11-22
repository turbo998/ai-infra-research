# Enterprise KV Cache Quantization Selection (`BF16` vs. `FP8 E4M3 Per-Head` vs. `KIVI Asymmetric INT4`) (`#489`)

Reproducible analytical model comparing uncompressed `BF16` KV Cache (`327.68 KB/tok`), Naive Per-Token `INT4` KV (`68.4%` accuracy due to Key channel outliers), `FP8 E4M3` Per-Head KV (`163.84 KB/tok`, `98.9%` accuracy), and `KIVI` Asymmetric `INT4` KV (Per-Channel Key + Per-Token Value + 128-token FP16 Residual Window = `90.1 KB/tok`, `98.4%` accuracy, `3.57x` higher concurrency).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_kv_cache_quantization_fp8_e4m3_int4_kivi_outlier_sizing_selection/run.py --check`
