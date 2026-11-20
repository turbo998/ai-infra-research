from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_email_ticket_triage_attachment_routing_automation_architecture_selection",
        "title": "Enterprise Shared-Mailbox & Ticket Triage Automation (500K Emails/Mo) Architecture Selection Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_inbound_emails": 500000,
            "avg_raw_thread_tokens": 6800,
            "avg_stripped_delta_tokens": 1768,
            "attachment_rate_pct": 42.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_raw_thread_frontier_llm_naive_autoreply",
                "name": "Tier 1: Raw Unstripped Thread to Frontier LLM + Unchecked Auto-Reply",
                "triage_routing_accuracy_pct": 88.5,
                "duplicate_ticket_rate_pct": 14.5,
                "auto_reply_loop_incident_rate_pct": 3.8,
                "prompt_token_reduction_pct": 0.0,
                "monthly_total_tco_usd": 168000,
                "verdict": "Anti-Pattern: 74% wasted quote tokens, infinite OOO auto-reply loops & $168K/mo bill",
            },
            {
                "tier_id": "tier2_legacy_regex_subject_rules",
                "name": "Tier 2: Legacy Subject-Regex Rules + Naive Thread Splitting",
                "triage_routing_accuracy_pct": 69.0,
                "duplicate_ticket_rate_pct": 18.2,
                "auto_reply_loop_incident_rate_pct": 1.2,
                "prompt_token_reduction_pct": 0.0,
                "monthly_total_tco_usd": 124000,
                "verdict": "31% misrouted tickets & 18.2% duplicate tickets when colleagues Reply-All",
            },
            {
                "tier_id": "tier3_sync_inline_vlm_every_attachment",
                "name": "Tier 3: Synchronous Frontier VLM on Every Email Page & Attachment",
                "triage_routing_accuracy_pct": 93.2,
                "duplicate_ticket_rate_pct": 6.4,
                "auto_reply_loop_incident_rate_pct": 0.8,
                "prompt_token_reduction_pct": 25.0,
                "monthly_total_tco_usd": 142000,
                "verdict": "18.4s webhook lag & excessive VLM cost across routine PDF/Excel attachments",
            },
            {
                "tier_id": "tier4_rfc3834_loop_guard_quote_strip_tiered_triage",
                "name": "Tier 4: RFC 3834 Loop Guard + References/SimHash Dedup + Quote Stripper + Tiered SLM/VLM",
                "triage_routing_accuracy_pct": 96.8,
                "duplicate_ticket_rate_pct": 0.2,
                "auto_reply_loop_incident_rate_pct": 0.0,
                "prompt_token_reduction_pct": 74.0,
                "monthly_total_tco_usd": 19500,
                "verdict": "Recommended: 96.8% accuracy, 0.0% reply loops, -74% tokens, 0.2% dupes & -88.4% TCO",
            },
        ],
        "headline_findings": {
            "triage_accuracy_pct": 96.8,
            "token_reduction_from_quote_stripping_pct": 74.0,
            "duplicate_ticket_rate_pct": 0.2,
            "tco_savings_vs_tier1_pct": 88.4,
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
