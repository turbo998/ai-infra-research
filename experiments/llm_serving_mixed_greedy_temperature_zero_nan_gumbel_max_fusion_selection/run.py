from __future__ import annotations
import json
from pathlib import Path


def evaluate_mixed_greedy_temperature_gumbel() -> dict:
    # Workload: Batch=128 (V=152,064) mixing 50% Greedy (T=0.0) and 50% Stochastic (T=0.7..1.2) requests
    tiers = [
        {
            "tier_id": "tier1_clamp_temp_1e5_div_softmax_multinomial_inf_nan_crash",
            "name": "Clamp `T=0 -> 1e-5` + In-Place `logits /= T` + `softmax` + `torch.multinomial`",
            "sampler_latency_ms": 3.45,
            "fp16_bf16_nan_crash_rate_pct": 18.5,
            "hbm_traffic_per_step_mb": 232.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Dividing logits by 1e-5 in FP16/BF16 overflows (>65504) or underflows ties -> softmax produces NaN -> multinomial CUDA assert crash!"
        },
        {
            "tier_id": "tier2_host_split_gather_argmax_and_multinomial_scatter",
            "name": "Split Batch Into Greedy vs Stochastic Sub-Batches (`index_select` + `argmax` + `multinomial` + `scatter`)",
            "sampler_latency_ms": 2.85,
            "fp16_bf16_nan_crash_rate_pct": 0.0,
            "hbm_traffic_per_step_mb": 195.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Avoids T=0 division, but dynamic boolean index_select/scatter breaks CUDAGraph capture and reads/writes 195MB across 6 kernels!"
        },
        {
            "tier_id": "tier3_fp32_upcast_softmax_plus_torch_where_argmax_override",
            "name": "Full Batch FP32 Upcast Softmax + `torch.where(is_greedy, argmax_ids, sampled_ids)`",
            "sampler_latency_ms": 1.68,
            "fp16_bf16_nan_crash_rate_pct": 0.0,
            "hbm_traffic_per_step_mb": 156.0,
            "action": "Confirm",
            "status": "pass",
            "note": "CUDAGraph-safe via static torch.where, but still computes both full Softmax+Multinomial AND ArgMax on every row (1.68ms)!"
        },
        {
            "tier_id": "tier4_single_pass_multiplicative_gumbel_max_argmax_triton_kernel",
            "name": "Single-Pass Multiplicative Gumbel-Max `argmax(logit_v - T_b * log(-log(U_v)))` Triton Kernel",
            "sampler_latency_ms": 0.38,
            "fp16_bf16_nan_crash_rate_pct": 0.0,
            "hbm_traffic_per_step_mb": 38.9,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Multiplies Gumbel noise by T_b instead of dividing logits by T_b! At T_b=0.0, T_b*G_v==0 (exact ArgMax); reads logits only ONCE (0.38ms, 7.5x faster)!"
        },
    ]
    return {
        "experiment_id": "llm_serving_mixed_greedy_temperature_zero_nan_gumbel_max_fusion_selection",
        "round": 320,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "batch_size": 128,
            "vocab_size_v": 152064,
            "traffic_mix": "50% Greedy (T=0.0) + 50% Stochastic (T=0.7..1.2)"
        },
        "tiers": tiers,
        "summary": {
            "latency_speedup_vs_split_x": round(2.85 / 0.38, 2),
            "hbm_traffic_reduction_x": round(232.0 / 38.9, 2),
            "nan_crash_rate_pct": 0.0,
            "recommended_tier": "tier4_single_pass_multiplicative_gumbel_max_argmax_triton_kernel"
        }
    }


if __name__ == "__main__":
    data = evaluate_mixed_greedy_temperature_gumbel()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
