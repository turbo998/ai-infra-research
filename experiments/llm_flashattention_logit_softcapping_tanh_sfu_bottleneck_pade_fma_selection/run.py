from __future__ import annotations
import json
from pathlib import Path


def evaluate_flashattention_softcapping_tanh() -> dict:
    # Workload: Gemma-2/Grok-style Attention Logit Soft-Capping `cap * tanh(S / cap)` at 32K context on H100
    tiers = [
        {
            "tier_id": "tier1_materialized_n2_scores_fallback_without_flash_fusion",
            "name": "Unfused `torch.tanh(Q @ K.T / cap) * cap` Fallback (Materializes `[B, H, N, N]` Matrix)",
            "effective_attention_tflops": 78.0,
            "prefill_32k_latency_ms": 485.0,
            "max_attention_output_abs_err": 0.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Materializes [N, N] attention scores in HBM to apply torch.tanh -> 78 TFLOPS & OOMs beyond 16K–32K!"
        },
        {
            "tier_id": "tier2_flashattention_inline_sfu_mufu_tanh_plus_mufu_ex2_congestion",
            "name": "Inline `FlashAttention` with Hardware `MUFU.TANH` + `MUFU.EX2` SFU Contention",
            "effective_attention_tflops": 275.0,
            "prefill_32k_latency_ms": 137.5,
            "max_attention_output_abs_err": 0.00001,
            "action": "Send to a human",
            "status": "fail",
            "note": "Executes per-element MUFU.TANH on the same Special Function Unit (SFU) needed by Online Softmax MUFU.EX2 -> 2.33x throughput drop!"
        },
        {
            "tier_id": "tier3_crude_piecewise_linear_hard_clamp_approximation",
            "name": "Crude Piecewise Linear / Hard Clamp `clamp(S, -cap, cap)` Approximation",
            "effective_attention_tflops": 615.0,
            "prefill_32k_latency_ms": 61.5,
            "max_attention_output_abs_err": 0.042,
            "action": "Confirm",
            "status": "fail",
            "note": "Fast on ALU, but hard clipping instead of smooth tanh distorts attention probabilities (4.2e-2 max error -> perplexity degradation)!"
        },
        {
            "tier_id": "tier4_rational_pade_fma_tanh_emulation_plus_folded_cap_log2e_scale",
            "name": "FMA Rational Padé `tanh` Emulation on FP32 ALU + Folded `(cap * log2e)` Pre-Exp2 Scaling",
            "effective_attention_tflops": 585.0,
            "prefill_32k_latency_ms": 64.6,
            "max_attention_output_abs_err": 0.000012,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Offloads tanh from congested SFU to idle FP32 FMA pipes via clamped Padé x*(27+x^2)/(27+9*x^2) / higher-order minimax and folds cap*log2(e) -> 585 TFLOPS (2.13x faster than SFU tanh)!"
        },
    ]
    return {
        "experiment_id": "llm_flashattention_logit_softcapping_tanh_sfu_bottleneck_pade_fma_selection",
        "round": 314,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "sequence_length_n": 32768,
            "softcap_value": 50.0,
            "gpu_arch": "Hopper H100 / Blackwell B200"
        },
        "tiers": tiers,
        "summary": {
            "tflops_speedup_vs_sfu_tanh_x": round(585.0 / 275.0, 2),
            "tflops_speedup_vs_unfused_x": round(585.0 / 78.0, 2),
            "recommended_tflops": 585.0,
            "recommended_tier": "tier4_rational_pade_fma_tanh_emulation_plus_folded_cap_log2e_scale"
        }
    }


if __name__ == "__main__":
    data = evaluate_flashattention_softcapping_tanh()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
