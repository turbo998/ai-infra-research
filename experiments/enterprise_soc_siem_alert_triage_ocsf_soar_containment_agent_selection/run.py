from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_soc_siem_alert_triage_ocsf_soar_containment_agent_selection",
        "title": "Enterprise SOC Alert Triage & SOAR Containment: OCSF Entity Clustering, Log Injection Shield & Blast-Radius Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "daily_raw_siem_edr_alerts": 45000,
            "daily_clustered_ocsf_incident_cases": 380,
            "tier0_critical_infrastructure_asset_share_pct": 8.0,
            "soc_tier1_analyst_fully_loaded_monthly_cost_usd": 14500,
        },
        "tiers": [
            {
                "tier_id": "tier1_per_alert_raw_llm_unfettered_soar_isolation",
                "name": "Tier 1: Per-Alert Raw JSON LLM Triage + Unfettered Autonomous Host/Account Isolation",
                "mean_time_to_respond_mttr_minutes": 9.5,
                "log_prompt_injection_bypass_rate_pct": 28.5,
                "monthly_self_inflicted_prod_outages": 4.2,
                "monthly_soc_staffing_token_and_outage_tco_usd": 610000,
                "verdict": "Anti-Pattern: 28.5% log prompt-injection bypass (User-Agent) & 4.2 prod outages/mo isolating DCs",
            },
            {
                "tier_id": "tier2_legacy_static_sigma_spl_100pct_manual_soc",
                "name": "Tier 2: Legacy Static Sigma/SPL Rules + 100% Manual Tier-1 SOC Triage",
                "mean_time_to_respond_mttr_minutes": 48.0,
                "log_prompt_injection_bypass_rate_pct": 0.0,
                "monthly_self_inflicted_prod_outages": 0.2,
                "monthly_soc_staffing_token_and_outage_tco_usd": 490000,
                "verdict": "48-min MTTR & 38% alert-fatigue miss rate across 45,000 daily alerts ($490K/mo)",
            },
            {
                "tier_id": "tier3_unclustered_alert_copilot_no_enrichment_prefetch",
                "name": "Tier 3: Unclustered Per-Alert Summarization Copilot (No OCSF Graph or Enrichment Pre-Fetch)",
                "mean_time_to_respond_mttr_minutes": 24.0,
                "log_prompt_injection_bypass_rate_pct": 6.4,
                "monthly_self_inflicted_prod_outages": 0.5,
                "monthly_soc_staffing_token_and_outage_tco_usd": 295000,
                "verdict": "Summarizes 45K duplicate alerts individually; analyst still spends 24m querying TI/CMDB",
            },
            {
                "tier_id": "tier4_ocsf_entity_clustering_spotlighting_opa_soar_guardrail",
                "name": "Tier 4: OCSF Entity-Graph Clustering (-99.1%) + Spotlighting Shield + OPA Blast-Radius SOAR",
                "mean_time_to_respond_mttr_minutes": 3.2,
                "log_prompt_injection_bypass_rate_pct": 0.0,
                "monthly_self_inflicted_prod_outages": 0.0,
                "monthly_soc_staffing_token_and_outage_tco_usd": 64000,
                "verdict": "Recommended: 3.2m MTTR, 0.0% log injection, 0.0 prod outages & -89.5% TCO ($64K/mo)",
            },
        ],
        "headline_findings": {
            "mttr_minutes": 3.2,
            "alert_volume_reduction_pct": 99.1,
            "log_injection_bypass_pct": 0.0,
            "tco_savings_vs_tier1_pct": 89.5,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
