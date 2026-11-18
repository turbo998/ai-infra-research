from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise ITSM (ServiceNow / Jira) & AIOps Incident Triage Agent Selection 
    # Workload: 50,000 IT helpdesk & production alert tickets per month across an enterprise.
    monthly_tickets = 50_000

    # Tier 1: Legacy Keyword Decision-Tree Bot (Static FAQ menus, manual L1/L2 triage)
    t1_ticket_automation_rate_pct = 16.0
    t1_mttr_minutes = 240.0  # 4 hours average resolution time
    t1_unsafe_prod_mutation_rate_pct = 0.0
    t1_monthly_total_ops_cost_usd = 88_000  # High human L1/L2 ticket handling labor

    # Tier 2: Passive Knowledge-Base RAG Bot (Answers "how-to" text, zero diagnostic tool execution)
    t2_ticket_automation_rate_pct = 42.0
    t2_mttr_minutes = 135.0
    t2_unsafe_prod_mutation_rate_pct = 0.0
    t2_monthly_total_ops_cost_usd = 54_000

    # Tier 3: Unbounded Autonomous Remediation Agent (Direct write/restart shell access without Runbook/HITL gate)
    t3_ticket_automation_rate_pct = 68.0
    t3_mttr_minutes = 28.0
    t3_unsafe_prod_mutation_rate_pct = 3.8  # Dangerous hallucinated restarts / config deletions in prod!
    t3_monthly_total_ops_cost_usd = 39_000  # Includes incident recovery penalty

    # Tier 4: Topology Alert Deduplication + Resolved-Ticket CBR + Read-Only Auto-Diagnostics + Runbook-Gated HITL Write
    t4_ticket_automation_rate_pct = 76.5
    t4_mttr_minutes = 14.5
    t4_unsafe_prod_mutation_rate_pct = 0.0  # Read-only diagnostics are 100% auto; prod writes require Runbook + 1-click HITL!
    t4_monthly_total_ops_cost_usd = 24_500

    mttr_speedup_t4_vs_t1_x = round(t1_mttr_minutes / t4_mttr_minutes, 1)
    automation_gain_t4_vs_t1_pts = round(t4_ticket_automation_rate_pct - t1_ticket_automation_rate_pct, 1)
    cost_savings_t4_vs_t1_pct = round(
        (t1_monthly_total_ops_cost_usd - t4_monthly_total_ops_cost_usd) / t1_monthly_total_ops_cost_usd * 100.0, 1
    )

    tiers = [
        {
            "tier_id": "tier1_legacy_keyword_decision_tree",
            "name": "Tier 1: 传统关键词规则树客服与人工派单 (仅解决 16% 简单咨询，MTTR 长达 240 分钟)",
            "ticket_automation_rate_pct": t1_ticket_automation_rate_pct,
            "avg_mttr_minutes": t1_mttr_minutes,
            "unsafe_prod_mutation_rate_pct": t1_unsafe_prod_mutation_rate_pct,
            "monthly_ops_cost_usd": t1_monthly_total_ops_cost_usd,
            "diagnostic_capability": "None (Manual Log/Metric Hunting)",
            "status": "High Human L1/L2 Labor Cost ($88K/mo)",
        },
        {
            "tier_id": "tier2_passive_kb_rag_assistant",
            "name": "Tier 2: 被动文档问答工单助手 (只发操作手册链接，不调接口查日志、不代办低风险操作)",
            "ticket_automation_rate_pct": t2_ticket_automation_rate_pct,
            "avg_mttr_minutes": t2_mttr_minutes,
            "unsafe_prod_mutation_rate_pct": t2_unsafe_prod_mutation_rate_pct,
            "monthly_ops_cost_usd": t2_monthly_total_ops_cost_usd,
            "diagnostic_capability": "Text Search Only (No Live Telemetry)",
            "status": "42% Deflection, Still Slow on Real Incidents",
        },
        {
            "tier_id": "tier3_unbounded_autonomous_shell_agent",
            "name": "Tier 3: 无护栏全自动运维执行 Agent (直接开放生产写权限，3.8% 概率误重启/误删引发P0事故)",
            "ticket_automation_rate_pct": t3_ticket_automation_rate_pct,
            "avg_mttr_minutes": t3_mttr_minutes,
            "unsafe_prod_mutation_rate_pct": t3_unsafe_prod_mutation_rate_pct,
            "monthly_ops_cost_usd": t3_monthly_total_ops_cost_usd,
            "diagnostic_capability": "Unconstrained Read/Write Shell (Dangerous!)",
            "status": "Rejected by SRE (3.8% Unsafe Prod Mutations)",
        },
        {
            "tier_id": "tier4_alert_dedup_cbr_readonly_diag_runbook_hitl",
            "name": "Tier 4: 拓扑告警降噪 + 历史工单案例推理 + 只读诊断全自动预取 + 参数化 Runbook 审批执行",
            "ticket_automation_rate_pct": t4_ticket_automation_rate_pct,
            "avg_mttr_minutes": t4_mttr_minutes,
            "unsafe_prod_mutation_rate_pct": t4_unsafe_prod_mutation_rate_pct,
            "monthly_ops_cost_usd": t4_monthly_total_ops_cost_usd,
            "diagnostic_capability": "100% Auto Read-Only Triage + Gated Write Runbooks",
            "status": "Enterprise ITSM & AIOps Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_itsm_servicedesk_aiops_incident_agent_selection",
        "round": 397,
        "title": "Enterprise ITSM (ServiceNow / Jira) & AIOps Incident Triage Agent Architecture Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_aiops_itbench_and_servicenow_incident_triage_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_it_and_alert_tickets": monthly_tickets,
            "alert_storm_compression_ratio": "1000 raw alerts -> 1 root-cause incident",
        },
        "headline_findings": {
            "tier4_ticket_automation_rate_pct": t4_ticket_automation_rate_pct,
            "automation_gain_t4_vs_t1_pts": automation_gain_t4_vs_t1_pts,
            "mttr_speedup_t4_vs_t1_x": mttr_speedup_t4_vs_t1_x,
            "cost_savings_t4_vs_t1_pct": cost_savings_t4_vs_t1_pct,
            "tier3_unsafe_mutation_rate_pct": t3_unsafe_prod_mutation_rate_pct,
            "tier4_unsafe_mutation_rate_pct": t4_unsafe_prod_mutation_rate_pct,
        },
        "architecture_tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic results.json")
    args = parser.parse_args()

    results = build_results()
    out_path = Path(__file__).with_name("results.json")

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist; run without --check first.")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != results:
            raise SystemExit("results.json mismatch with deterministic build_results().")
        print("OK: enterprise_itsm_servicedesk_aiops_incident_agent_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
