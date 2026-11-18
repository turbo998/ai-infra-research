# Enterprise LLM Quantization (BF16 vs. FP8 W8A8 vs. INT4 AWQ) & Hardware Matching Evaluator 

Deterministic analytical evaluator comparing four enterprise quantization schemes for a 70B LLM (`4,096` input / `512` output tokens, `200 RPS` peak):

1. **Tier 1 (`tier1_unquantized_bf16_w16a16kv16`)**: Unquantized BF16 Weights + BF16 KV (`86.0` composite score [`100.0%` retention], `140.0 GB` weight VRAM, `340 tok/s/GPU`, `$48,000/mo`).
2. **Tier 2 (`tier2_naive_uncalibrated_int4_rtn`)**: Uncalibrated Naive Round-to-Nearest INT4 (`80.2` composite score [`93.3%` retention, `-5.8 pts` drop on financial math/SQL], `38.0 GB` weight VRAM, `590 tok/s/GPU`, `$16,500/mo`).
3. **Tier 3 (`tier3_calibrated_awq_gptq_marlin_w4a16`)**: Domain-Calibrated INT4 Weight-Only (`AWQ / GPTQ Marlin W4A16`, `84.2` score [`97.9%` retention], `38.5 GB` weight VRAM, `610 tok/s/GPU`, `$18,000/mo` — best for Ampere `A100/A10G` lacking native FP8 Tensor Cores).
4. **Tier 4 (`tier4_native_fp8_w8a8_plus_fp8_kv_cache`)**: Hardware-Native `FP8 W8A8` + `FP8 KV Cache` (`W8A8KV8` on Hopper `H100/H200` & Ada `L40S`, `85.7` score [`99.7%` retention, only `0.3 pt` delta], `70.0 GB` weight VRAM, `762 tok/s/GPU` [`2.24x` vs. BF16], `$21,400/mo` [`-55.4%` vs. BF16]).

## Usage

```bash
python experiments/enterprise_llm_quantization_bf16_fp8_int4_hardware_selection/run.py
python experiments/enterprise_llm_quantization_bf16_fp8_int4_hardware_selection/run.py --check
```
