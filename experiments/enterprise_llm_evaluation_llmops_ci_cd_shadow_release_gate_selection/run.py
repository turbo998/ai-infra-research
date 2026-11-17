from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise LLM Evaluation, LLMOps CI/CD Release Gate & Shadow Replay Selection (Round 380)
    # Workload: 25 enterprise AI applications, 120 prompt/model/RAG releases per month,
    # 500-case stratified Golden Dataset per CI run + 5% production shadow sampling.
    monthly_releases = 120
    golden_dataset_size = 500

    # Tier 1: Manual Spreadsheet Vibe-Check (20 ad-hoc samples in Excel per release)
    t1_regression_catch_rate_pct = 38.0
    t1_human_kappa_agreement = 1.00  # Small manual sample, but massive blind spots
    t1_release_cycle_hours = 120.0  # 5 days of manual QA bottleneck
    t1_monthly_eval_cost_usd = 22_000  # High human QA engineering labor hours

    # Tier 2: Uncalibrated 1-5 Likert Scale Frontier LLM-as-a-Judge ("Score 1 to 5")
    t2_regression_catch_rate_pct = 68.5
    t2_human_kappa_agreement = 0.56  # Verbosity bias, position bias, central tendency (mostly 4/5)
    t2_release_cycle_hours = 1.2
    t2_monthly_eval_cost_usd = 14_800  # Expensive frontier judge API tokens across all runs

    # Tier 3: Deterministic Code Assertions + Decomposed Binary Rubric Judge (Promptfoo / DeepEval / Ragas)
    t3_regression_catch_rate_pct = 91.2
    t3_human_kappa_agreement = 0.86  # Atomic Yes/No checklists + pairwise swap debiasing
    t3_release_cycle_hours = 0.37  # ~22 minutes in CI
    t3_monthly_eval_cost_usd = 2_900

    # Tier 4: CI/CD Stratified Gate + Production Shadow Traffic Replay + Automated Bad-Case Flywheel
    t4_regression_catch_rate_pct = 98.6
    t4_human_kappa_agreement = 0.91
    t4_release_cycle_hours = 0.30  # 18 minutes automated CI + shadow canary
    t4_monthly_eval_cost_usd = 3_400

    catch_rate_gain_t4_vs_t1_pts = round(t4_regression_catch_rate_pct - t1_regression_catch_rate_pct, 1)
    kappa_gain_t4_vs_t2 = round(t4_human_kappa_agreement - t2_human_kappa_agreement, 2)
    cost_savings_t4_vs_t2_pct = round((t2_monthly_eval_cost_usd - t4_monthly_eval_cost_usd) / t2_monthly_eval_cost_usd * 100.0, 1)
    release_speedup_t4_vs_t1_x = round(t1_release_cycle_hours / t4_release_cycle_hours, 1)

    tiers = [
        {
            "tier_id": "tier1_manual_spreadsheet_vibecheck",
            "name": "Tier 1: 人工 Excel 盲测抽检 (每次发版人工肉眼看 20 条样例，“凭感觉上线”)",
            "pre_prod_regression_catch_rate_pct": t1_regression_catch_rate_pct,
            "human_expert_kappa_agreement": t1_human_kappa_agreement,
            "release_gate_latency_hours": t1_release_cycle_hours,
            "monthly_eval_cost_usd": t1_monthly_eval_cost_usd,
            "production_drift_coverage": "None (62% Regressions Leak to Prod)",
            "status": "Slowest & Prompt Whack-a-Mole",
        },
        {
            "tier_id": "tier2_uncalibrated_likert_llm_judge",
            "name": "Tier 2: 无校准 1~5 分制大模型裁判 (直接让顶配大模型给回答打 1~5 分)",
            "pre_prod_regression_catch_rate_pct": t2_regression_catch_rate_pct,
            "human_expert_kappa_agreement": t2_human_kappa_agreement,
            "release_gate_latency_hours": t2_release_cycle_hours,
            "monthly_eval_cost_usd": t2_monthly_eval_cost_usd,
            "production_drift_coverage": "Low Agreement (Kappa=0.56, Verbosity Bias)",
            "status": "Noisy Scores & High Token Bill",
        },
        {
            "tier_id": "tier3_deterministic_plus_binary_rubric",
            "name": "Tier 3: 确定性代码断言 + 二值化原子量表裁判 (Promptfoo / DeepEval / Ragas)",
            "pre_prod_regression_catch_rate_pct": t3_regression_catch_rate_pct,
            "human_expert_kappa_agreement": t3_human_kappa_agreement,
            "release_gate_latency_hours": t3_release_cycle_hours,
            "monthly_eval_cost_usd": t3_monthly_eval_cost_usd,
            "production_drift_coverage": "CI Golden Set Only (Static Offline Cases)",
            "status": "Standard Offline CI Baseline",
        },
        {
            "tier_id": "tier4_cicd_gate_shadow_replay_flywheel",
            "name": "Tier 4: CI/CD 黄金集门禁 + 生产影子流量回放 (Shadow Replay) + 坏例自动入库飞轮",
            "pre_prod_regression_catch_rate_pct": t4_regression_catch_rate_pct,
            "human_expert_kappa_agreement": t4_human_kappa_agreement,
            "release_gate_latency_hours": t4_release_cycle_hours,
            "monthly_eval_cost_usd": t4_monthly_eval_cost_usd,
            "production_drift_coverage": "Full Closed-Loop (Offline Golden + Live Shadow Canary)",
            "status": "Enterprise LLMOps Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_llm_evaluation_llmops_ci_cd_shadow_release_gate_selection",
        "round": 380,
        "title": "Enterprise LLM Evaluation, LLMOps CI/CD Release Gate & Shadow Traffic Replay Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_llm_as_a_judge_and_llmops_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "enterprise_ai_apps": 25,
            "monthly_releases": monthly_releases,
            "golden_dataset_cases_per_app": golden_dataset_size,
            "shadow_traffic_sampling_pct": 5.0,
        },
        "headline_findings": {
            "tier4_regression_catch_rate_pct": t4_regression_catch_rate_pct,
            "catch_rate_gain_t4_vs_t1_pts": catch_rate_gain_t4_vs_t1_pts,
            "tier4_human_kappa_agreement": t4_human_kappa_agreement,
            "kappa_gain_t4_vs_t2": kappa_gain_t4_vs_t2,
            "cost_savings_t4_vs_t2_pct": cost_savings_t4_vs_t2_pct,
            "release_speedup_t4_vs_t1_x": release_speedup_t4_vs_t1_x,
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
        print("OK: enterprise_llm_evaluation_llmops_ci_cd_shadow_release_gate_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
