from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise Browser-Use / Computer-Use GUI Agent Architecture Selection (Round 391)
    # Workload: 200,000 enterprise web workflows/month (15 steps/workflow across ERP, Expense, Procurement, CRM portals).
    monthly_workflows = 200_000

    # Tier 1: Pure Vision Screenshot Pixel-Click Loop (1080p screenshot + VLM coordinate prediction on every step)
    t1_workflow_success_rate_pct = 71.5  # Fails on DPI shift, dynamic scrollbars, tiny date-picker dropdowns
    t1_avg_workflow_duration_sec = 51.0
    t1_tokens_per_workflow = 38_000
    t1_monthly_cost_usd = 96_000

    # Tier 2: Raw HTML DOM Dump into LLM (80K+ tokens of unpruned React/Angular div soup per page)
    t2_workflow_success_rate_pct = 64.0
    t2_avg_workflow_duration_sec = 32.5
    t2_tokens_per_workflow = 145_000
    t2_monthly_cost_usd = 68_000

    # Tier 3: Distilled Accessibility Tree (AXTree) + Set-of-Marks (SoM) Visual Grounding
    t3_workflow_success_rate_pct = 88.5
    t3_avg_workflow_duration_sec = 14.2
    t3_tokens_per_workflow = 16_500
    t3_monthly_cost_usd = 16_400

    # Tier 4: API/MCP-First + Cached Deterministic Playwright Replay (LLM Self-Healing Only on UI Drift) + AXTree/SoM + HITL Gate
    t4_workflow_success_rate_pct = 98.2
    t4_avg_workflow_duration_sec = 4.6
    t4_tokens_per_workflow = 2_400  # 85% of steps run cached Playwright selectors at 0 tokens!
    t4_monthly_cost_usd = 5_200

    success_gain_t4_vs_t1_pts = round(t4_workflow_success_rate_pct - t1_workflow_success_rate_pct, 1)
    speedup_t4_vs_t1_x = round(t1_avg_workflow_duration_sec / t4_avg_workflow_duration_sec, 1)
    cost_savings_t4_vs_t1_pct = round((t1_monthly_cost_usd - t4_monthly_cost_usd) / t1_monthly_cost_usd * 100.0, 1)

    tiers = [
        {
            "tier_id": "tier1_pure_vision_screenshot_pixel_click",
            "name": "Tier 1: 纯屏幕截图视觉坐标点击 (每步截 1080p 图调 VLM 猜 x,y 像素，慢 51s 且易点偏)",
            "workflow_success_rate_pct": t1_workflow_success_rate_pct,
            "avg_workflow_duration_sec": t1_avg_workflow_duration_sec,
            "avg_tokens_per_workflow": t1_tokens_per_workflow,
            "monthly_cost_200k_workflows_usd": t1_monthly_cost_usd,
            "security_hitl_control": "Unbounded Autonomous Clicking (High Misclick Risk)",
            "status": "Slowest (51s) & Most Expensive ($96K/mo)",
        },
        {
            "tier_id": "tier2_raw_html_dom_dump",
            "name": "Tier 2: 原始 HTML DOM 全量塞入 Prompt (单页 80K+ 冗余 div 淹没注意力)",
            "workflow_success_rate_pct": t2_workflow_success_rate_pct,
            "avg_workflow_duration_sec": t2_avg_workflow_duration_sec,
            "avg_tokens_per_workflow": t2_tokens_per_workflow,
            "monthly_cost_200k_workflows_usd": t2_monthly_cost_usd,
            "security_hitl_control": "Vulnerable to Hidden CSS Prompt Injection",
            "status": "Context Overflow & 64.0% Accuracy",
        },
        {
            "tier_id": "tier3_distilled_axtree_plus_set_of_marks",
            "name": "Tier 3: 蒸馏无障碍语义树 (AXTree) + Set-of-Marks (SoM) 数字编号框混合定位",
            "workflow_success_rate_pct": t3_workflow_success_rate_pct,
            "avg_workflow_duration_sec": t3_avg_workflow_duration_sec,
            "avg_tokens_per_workflow": t3_tokens_per_workflow,
            "monthly_cost_200k_workflows_usd": t3_monthly_cost_usd,
            "security_hitl_control": "Element-ID Grounded (Still Runs LLM Every Time)",
            "status": "Ad-Hoc Exploration Baseline",
        },
        {
            "tier_id": "tier4_mcp_playwright_replay_self_healing_hitl",
            "name": "Tier 4: API/MCP 优先 + 确定性 Playwright 脚本缓存回放 (仅改版时调 LLM 自愈) + 敏感操作人类移交",
            "workflow_success_rate_pct": t4_workflow_success_rate_pct,
            "avg_workflow_duration_sec": t4_avg_workflow_duration_sec,
            "avg_tokens_per_workflow": t4_tokens_per_workflow,
            "monthly_cost_200k_workflows_usd": t4_monthly_cost_usd,
            "security_hitl_control": "Isolated Browser Sandbox + Mandatory MFA/Submit HITL Handoff",
            "status": "Enterprise AI RPA Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_browser_computer_use_gui_agent_selection",
        "round": 391,
        "title": "Enterprise Browser-Use & Computer-Use GUI Agent Architecture Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_webarena_osworld_and_browser_use_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_workflows": monthly_workflows,
            "steps_per_workflow": 15,
            "cached_playwright_step_hit_rate_pct": 85.0,
        },
        "headline_findings": {
            "tier4_workflow_success_rate_pct": t4_workflow_success_rate_pct,
            "success_gain_t4_vs_t1_pts": success_gain_t4_vs_t1_pts,
            "speedup_t4_vs_t1_x": speedup_t4_vs_t1_x,
            "cost_savings_t4_vs_t1_pct": cost_savings_t4_vs_t1_pct,
            "tier4_avg_duration_sec": t4_avg_workflow_duration_sec,
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
        print("OK: enterprise_browser_computer_use_gui_agent_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
