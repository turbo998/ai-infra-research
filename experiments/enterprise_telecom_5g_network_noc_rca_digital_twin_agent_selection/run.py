from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_telecom_5g_network_noc_rca_digital_twin_agent_selection_v1",
        "title": "Enterprise Telecom 5G RAN/Core NOC Alarm Storm RCA, Multi-Vendor Netconf/YANG & Digital-Twin Canary Rollback Agent Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "managed_5g_gnodeb_base_stations": 45000,
            "metro_ip_and_5g_core_upf_nodes": 1200,
            "daily_raw_network_alarms": 1200000,
            "sla_target": "Cross-layer root-cause accuracy >= 94%, network config outage rate == 0.0%, MTTR <= 10.0 minutes (TM Forum Autonomous Network L4)",
        },
        "tiers": [
            {
                "tier_id": "tier1_legacy_noc_static_rules_and_manual_cli_triage",
                "name": "Tier 1: Legacy Static NOC Alarm Correlation Rules + Manual Engineer CLI Triage",
                "cross_layer_root_cause_localization_accuracy_pct": 64.5,
                "config_change_network_outage_or_syntax_error_rate_pct": 1.8,
                "mean_time_to_restore_service_minutes": 95.0,
                "monthly_noc_ops_and_sla_penalty_cost_usd": 340000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_raw_alarm_stream_llm_with_direct_netconf_cli_push",
                "name": "Tier 2: Raw Alarm Stream Stuffed into LLM with Direct Unverified CLI/Netconf Push",
                "cross_layer_root_cause_localization_accuracy_pct": 56.0,
                "config_change_network_outage_or_syntax_error_rate_pct": 13.6,
                "mean_time_to_restore_service_minutes": 38.0,
                "monthly_noc_ops_and_sla_penalty_cost_usd": 510000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_3gpp_vendor_manual_rag_plus_ticket_copilot",
                "name": "Tier 3: Standard 3GPP / Vendor Manual Vector RAG + Read-Only Ticket Copilot",
                "cross_layer_root_cause_localization_accuracy_pct": 78.5,
                "config_change_network_outage_or_syntax_error_rate_pct": 4.2,
                "mean_time_to_restore_service_minutes": 45.0,
                "monthly_noc_ops_and_sla_penalty_cost_usd": 155000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_cross_layer_topology_graph_rca_plus_yang_digital_twin_canary_agent",
                "name": "Tier 4: Cross-Layer Topology Causal Graph RCA + Netconf/YANG Compiler + Network Digital-Twin Simulation & 60s Canary Rollback Agent",
                "cross_layer_root_cause_localization_accuracy_pct": 95.8,
                "config_change_network_outage_or_syntax_error_rate_pct": 0.0,
                "mean_time_to_restore_service_minutes": 8.2,
                "monthly_noc_ops_and_sla_penalty_cost_usd": 49000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "Metro DWDM Optical Fiber Cut Triggering 42,000 Sympathetic IP/5G gNodeB Alarms + Multi-Vendor (Cisco IOS-XR / Huawei VRP / Nokia SR-OS) BGP Traffic Reroute",
            "tier2_raw_llm_rca_accuracy_pct": 44.0,
            "tier4_topology_causal_graph_rca_accuracy_pct": 95.2,
            "rca_gain_pts": 51.2,
            "tier2_outage_rate_pct": 13.6,
            "tier4_digital_twin_outage_rate_pct": 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = build_payload()
    out_path = Path(__file__).resolve().parent / "results.json"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            return 1
        return 0 if out_path.read_text(encoding="utf-8") == rendered else 1

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
