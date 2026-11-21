from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    enterprise_tools = 250
    tiers = [
        {
            "tier": "Tier 1: Stuff All 250 MCP Tool Schemas in System Prompt + Serial ReAct + Shared Static Key",
            "tokens_per_5step_task": 285000,
            "tool_selection_accuracy_pct": 64.2,
            "prefix_kv_hit_rate_pct": 88.0,
            "monthly_mcp_token_tco_usd": 142500,
        },
        {
            "tier": "Tier 2: Cloud API Gateway Managed MCP (Azure APIM MCP Server + OAuth 2.1 OBO + Domain Grouping)",
            "tokens_per_5step_task": 56800,
            "tool_selection_accuracy_pct": 88.5,
            "prefix_kv_hit_rate_pct": 89.2,
            "monthly_mcp_token_tco_usd": 28400,
        },
        {
            "tier": "Tier 3: Naive Per-Turn System-Prompt Tool-RAG Replacement (Breaks Prefix KV Cache!)",
            "tokens_per_5step_task": 42000,
            "tool_selection_accuracy_pct": 92.4,
            "prefix_kv_hit_rate_pct": 18.5,
            "monthly_mcp_token_tco_usd": 36800,
        },
        {
            "tier": "Tier 4: APIM/Envoy OBO Gateway + Tail-Appended Top-8 Deferred Discovery + Code-Mode + JSON Projection",
            "tokens_per_5step_task": 16400,
            "tool_selection_accuracy_pct": 96.8,
            "prefix_kv_hit_rate_pct": 91.5,
            "monthly_mcp_token_tco_usd": 8200,
        },
    ]
    return {
        "experiment_id": "enterprise_mcp_gateway_apim_envoy_semantic_tool_router_obo_auth_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "registered_mcp_tools": enterprise_tools,
            "tokens_per_tool_schema": 220,
            "monthly_agent_tasks_millions": 20.0,
            "raw_erp_json_response_kb": 120,
            "projected_json_response_kb": 1.2,
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
        print("OK: enterprise_mcp_gateway_apim_envoy_semantic_tool_router_obo_auth_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
