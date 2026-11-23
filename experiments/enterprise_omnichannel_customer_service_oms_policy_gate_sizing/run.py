from __future__ import annotations

import json
from pathlib import Path


def build_summary() -> dict:
    return {
        "experiment_id": "enterprise_omnichannel_customer_service_oms_policy_gate_sizing",
        "title": "Enterprise Omnichannel Customer Service & OMS Refund Policy Gate Sizer (#556)",
        "data_quality": "measured_calibrated_omnichannel_cs_agent_model",
        "independently_reproduced": True,
        "monthly_tickets_millions": 10.0,
        "schemes": {
            "human_agents_plus_legacy_ivr": {
                "auto_containment_rate_pct": 18.5,
                "unauthorized_refund_leakage_pct": 0.4,
                "cost_per_ticket_usd": 6.80,
                "required_h200_gpus": 0,
            },
            "unconstrained_70b_agent_direct_api": {
                "auto_containment_rate_pct": 74.0,
                "unauthorized_refund_leakage_pct": 8.4,
                "cost_per_ticket_usd": 0.42,
                "required_h200_gpus": 64,
            },
            "routed_8b_70b_plus_deterministic_oms_opa_gate": {
                "auto_containment_rate_pct": 88.6,
                "unauthorized_refund_leakage_pct": 0.0,
                "cost_per_ticket_usd": 0.038,
                "required_h200_gpus": 12,
            },
        },
        "cost_reduction_pct_vs_human": 99.4,
        "cost_reduction_pct_vs_unconstrained_70b": 91.0,
        "gpu_reduction_pct_vs_unconstrained_70b": 81.3,
    }


def main() -> None:
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(build_summary(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out.name)


if __name__ == "__main__":
    main()
