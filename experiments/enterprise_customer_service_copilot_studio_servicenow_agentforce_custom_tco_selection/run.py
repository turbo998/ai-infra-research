from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    monthly_conversations_m = 3.0
    tiers = [
        {
            "tier": "Tier 1: 100% High-Ticket CRM Native SaaS Agent (Agentforce / ServiceNow Now Assist for All Public Traffic)",
            "autonomous_deflection_rate_pct": 76.0,
            "human_aht_minutes": 3.5,
            "blended_cost_per_conversation_usd": 0.45,
            "monthly_platform_tco_usd": 1350000,
        },
        {
            "tier": "Tier 2: 100% Microsoft Copilot Studio (Managed Low-Code + 1,400+ Enterprise Connectors)",
            "autonomous_deflection_rate_pct": 79.5,
            "human_aht_minutes": 3.2,
            "blended_cost_per_conversation_usd": 0.12,
            "monthly_platform_tco_usd": 360000,
        },
        {
            "tier": "Tier 3: 100% DIY Pro-Code (LangGraph + vLLM + Custom-Built ERP/CRM Connectors)",
            "autonomous_deflection_rate_pct": 78.5,
            "human_aht_minutes": 3.6,
            "blended_cost_per_conversation_usd": 0.022,
            "monthly_platform_tco_usd": 66000,
        },
        {
            "tier": "Tier 4: 82/18 Hybrid Architecture (Front-Door AI Search/vLLM 82% + Copilot Studio/CRM Action Connectors 13% + Warm Human Handoff 5%)",
            "autonomous_deflection_rate_pct": 86.4,
            "human_aht_minutes": 2.8,
            "blended_cost_per_conversation_usd": 0.018,
            "monthly_platform_tco_usd": 54000,
        },
    ]
    return {
        "experiment_id": "enterprise_customer_service_copilot_studio_servicenow_agentforce_custom_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "monthly_conversations_millions": monthly_conversations_m,
            "front_door_faq_readonly_share": 0.82,
            "transactional_connector_share": 0.13,
            "human_escalation_share": 0.05,
        },
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_customer_service_copilot_studio_servicenow_agentforce_custom_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
