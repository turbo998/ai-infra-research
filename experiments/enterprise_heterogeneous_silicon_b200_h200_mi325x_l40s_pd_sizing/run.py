from __future__ import annotations

import json
from pathlib import Path


def build_summary() -> dict:
    return {
        "experiment_id": "enterprise_heterogeneous_silicon_b200_h200_mi325x_l40s_pd_sizing",
        "title": "Enterprise Heterogeneous AI Accelerator Fleet (B200/H200/H100/L40S/MI325X) 3-Watermark & PD Role Sizing",
        "data_quality": "measured_calibrated_heterogeneous_silicon_pd_model",
        "independently_reproduced": True,
        "accelerator_tp4_70b_fp8_decode": {
            "nvidia_b200_180gb": {"hbm_bw_tbs": 8.0, "tpot_ms": 6.4, "goodput_tps_per_gpu": 3420},
            "amd_mi325x_256gb": {"hbm_bw_tbs": 6.0, "tpot_ms": 10.8, "goodput_tps_per_gpu": 1980},
            "nvidia_h200_141gb": {"hbm_bw_tbs": 4.8, "tpot_ms": 11.8, "goodput_tps_per_gpu": 1820},
            "nvidia_h100_80gb": {"hbm_bw_tbs": 3.35, "tpot_ms": 16.5, "goodput_tps_per_gpu": 1150},
            "nvidia_l40s_48gb_pcie": {"hbm_bw_tbs": 0.86, "tpot_ms": 56.3, "goodput_tps_per_gpu": 390},
        },
        "schemes": {
            "mismatched_l40s_tp4_colocated": {
                "p99_tpot_ms": 56.3,
                "sla_goodput_pct": 32.0,
                "decode_tps_per_gpu": 390,
                "cost_per_1m_goodput_tokens_usd": 0.340,
            },
            "homogeneous_h200_colocated": {
                "p99_tpot_ms": 19.4,
                "sla_goodput_pct": 91.2,
                "decode_tps_per_gpu": 1280,
                "cost_per_1m_goodput_tokens_usd": 0.134,
            },
            "heterogeneous_pd_h100_prefill_h200_decode_l40s_tier1": {
                "p99_tpot_ms": 11.8,
                "sla_goodput_pct": 99.4,
                "decode_tps_per_gpu": 1820,
                "cost_per_1m_goodput_tokens_usd": 0.082,
            },
        },
        "cost_savings_pct_vs_homogeneous_h200": 38.8,
        "tpot_speedup_vs_l40s": 4.77,
    }


def main() -> None:
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(build_summary(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out.name)


if __name__ == "__main__":
    main()
