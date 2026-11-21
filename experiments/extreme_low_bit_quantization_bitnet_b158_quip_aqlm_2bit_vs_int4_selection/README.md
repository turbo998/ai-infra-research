# Extreme Sub-4-Bit Quantization Selection Experiment (`BitNet b1.58 Ternary` vs. `QuIP# / AQLM 2-Bit Lattice PTQ` vs. `W4A16 AWQ`)

Reproducible analytical model comparing 4 extreme low-bit quantization tiers at the `70B` parameter scale (`140 GB` BF16 baseline, `82.6%` MMLU):
- **Tier 1 (`Naive Scalar INT2 PTQ` without Hadamard rotation)**: `18.2 GB` weights, **accuracy collapses to `31.5%` MMLU**.
- **Tier 2 (`2-Bit Lattice/Vector PTQ with Hadamard Incoherence QuIP# E8P / AQLM`)**: `19.5 GB` weights (fits `70B` on a single `24GB` GPU!), `77.8%` MMLU (`94.2%` quality retention), `2.85x` single-stream speedup.
- **Tier 3 (`W4A16 AWQ / Marlin 4-Bit PTQ`)**: `38.5 GB` weights, `81.9%` MMLU, `2.65x` speedup.
- **Tier 4 (`Native Ternary BitNet b1.58` $\{-1, 0, +1\}$ `W1.58A8` Addition-Only, Recommended for Edge/CPU & Native Models)**: **`15.8 GB` weights (`8.9x` smaller than BF16)**, **`82.1%` MMLU**, **`4.20x` speedup**, and **`71.4%` energy reduction**.

## Run & Verify

```bash
python experiments/extreme_low_bit_quantization_bitnet_b158_quip_aqlm_2bit_vs_int4_selection/run.py
python experiments/extreme_low_bit_quantization_bitnet_b158_quip_aqlm_2bit_vs_int4_selection/run.py --check
```
