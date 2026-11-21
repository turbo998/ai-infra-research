from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "agentic_sandbox_egress_firewall_secret_broker_exfiltration_prevention_selection",
        "title": "Agentic Sandbox Egress Firewall & Secret Broker Selection: Zero-Trust Out-of-Sandbox Credential Injection vs. Raw Env Vars & L3 Allowlists",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "threat_vectors": "Indirect Prompt Injection (Markdown Image Exfil, DNS Tunneling, print(os.environ), Allowed-Domain GitHub/S3 POST Exfil)",
            "agent_type": "Enterprise Coding Agent & Tool-Using Workflow Sandbox",
        },
        "tiers": [
            {
                "tier": "Tier 1: Unrestricted Outbound Egress + Raw API Keys in Container os.environ",
                "data_exfiltration_block_rate_pct": 1.5,
                "raw_credential_theft_risk_pct": 100.0,
                "added_egress_proxy_latency_ms": 0.0,
                "blocks_allowed_domain_post_exfil": False,
                "recommended_scenario": "Critical vulnerability (never deploy in production)",
            },
            {
                "tier": "Tier 2: L3/L4 IP & DNS Allowlist (Cilium NetworkPolicy) + Raw Secrets in os.environ",
                "data_exfiltration_block_rate_pct": 58.0,
                "raw_credential_theft_risk_pct": 100.0,
                "added_egress_proxy_latency_ms": 0.2,
                "blocks_allowed_domain_post_exfil": False,
                "recommended_scenario": "Insufficient (attackers exfiltrate os.environ secrets via allowed github.com / s3 domains)",
            },
            {
                "tier": "Tier 3: L7 Forward Proxy (Envoy/Squid Method + Path ACL) + Short-Lived Scoped Tokens",
                "data_exfiltration_block_rate_pct": 91.5,
                "raw_credential_theft_risk_pct": 25.0,
                "added_egress_proxy_latency_ms": 1.4,
                "blocks_allowed_domain_post_exfil": True,
                "recommended_scenario": "Baseline enterprise sandbox with repo-scoped fine-grained tokens",
            },
            {
                "tier": "Tier 4 (Recommended): Out-of-Sandbox L7 Secret Broker (Zero Raw Keys in Sandbox) + Path/Method ACL + CSP img-src Sinkhole",
                "data_exfiltration_block_rate_pct": 99.8,
                "raw_credential_theft_risk_pct": 0.0,
                "added_egress_proxy_latency_ms": 1.8,
                "blocks_allowed_domain_post_exfil": True,
                "recommended_scenario": "Production Coding Agents & MCP Sandboxes (0% raw token exposure, 99.8% exfil blocked)",
            },
        ],
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
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
