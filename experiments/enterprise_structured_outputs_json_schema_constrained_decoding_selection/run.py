from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise Structured Outputs, JSON Schema Constrained Decoding & API Contracts 
    # Workload: 20,000,000 structured microservice & ERP API extraction/decision calls per month (128K vocab LLM).
    monthly_calls = 20_000_000

    # Tier 1: Prompt-Only Soft JSON Instruction + Regex Strip + Retry Loop
    t1_json_syntax_pass_rate_pct = 90.6  # 9.4% first-pass JSON parse failures (unescaped quotes, trailing commas)
    t1_business_semantic_accuracy_pct = 81.2
    t1_per_token_mask_overhead_ms = 0.0
    t1_p99_e2e_latency_ms = 4_850.0  # Spikes due to multi-turn JSON repair retries
    t1_monthly_cost_usd = 18_600

    # Tier 2: Legacy Uncached CPU Interpreter CFG Masking (Per-token vocab mask computed on CPU)
    t2_json_syntax_pass_rate_pct = 100.0
    t2_business_semantic_accuracy_pct = 84.5  # Decision field placed BEFORE reasoning field hurts accuracy!
    t2_per_token_mask_overhead_ms = 28.0
    t2_p99_e2e_latency_ms = 3_420.0
    t2_monthly_cost_usd = 24_200

    # Tier 3: Pre-Compiled Zero-Overhead Constrained Decoding (XGrammar / llguidance / Outlines with Naive Schema)
    t3_json_syntax_pass_rate_pct = 100.0
    t3_business_semantic_accuracy_pct = 84.5  # Still suffers when decision_enum is placed before reasoning_steps!
    t3_per_token_mask_overhead_ms = 0.04
    t3_p99_e2e_latency_ms = 1_180.0
    t3_monthly_cost_usd = 11_200

    # Tier 4: Reasoning-First Flattened Schema + Pre-Cached XGrammar/llguidance + Pydantic Semantic Validator Gate
    t4_json_syntax_pass_rate_pct = 100.0
    t4_business_semantic_accuracy_pct = 97.8  # +13.3 pts gain by ordering rationale_summary BEFORE decision_enum!
    t4_per_token_mask_overhead_ms = 0.03
    t4_p99_e2e_latency_ms = 960.0
    t4_monthly_cost_usd = 9_800

    semantic_gain_t4_vs_t3_pts = round(t4_business_semantic_accuracy_pct - t3_business_semantic_accuracy_pct, 1)
    latency_speedup_t4_vs_t1_x = round(t1_p99_e2e_latency_ms / t4_p99_e2e_latency_ms, 2)
    cost_savings_t4_vs_t1_pct = round((t1_monthly_cost_usd - t4_monthly_cost_usd) / t1_monthly_cost_usd * 100.0, 1)

    tiers = [
        {
            "tier_id": "tier1_prompt_only_regex_retry",
            "name": "Tier 1: Prompt 软提醒 + 正则清洗 + 报错重试 (首轮 9.4% JSON 解析失败，P99 延迟飙至 4.85s)",
            "json_syntax_valid_pct": t1_json_syntax_pass_rate_pct,
            "business_semantic_accuracy_pct": t1_business_semantic_accuracy_pct,
            "per_token_grammar_overhead_ms": t1_per_token_mask_overhead_ms,
            "p99_e2e_latency_ms": t1_p99_e2e_latency_ms,
            "monthly_cost_20m_calls_usd": t1_monthly_cost_usd,
            "status": "Unreliable for Microservice Contracts",
        },
        {
            "tier_id": "tier2_legacy_cpu_cfg_masking",
            "name": "Tier 2: 传统未缓存 CPU 语法树逐词掩码 (每吐1个词卡顿 28ms，拖垮 GPU 批处理吞吐)",
            "json_syntax_valid_pct": t2_json_syntax_pass_rate_pct,
            "business_semantic_accuracy_pct": t2_business_semantic_accuracy_pct,
            "per_token_mask_overhead_ms": t2_per_token_mask_overhead_ms,
            "p99_e2e_latency_ms": t2_p99_e2e_latency_ms,
            "monthly_cost_20m_calls_usd": t2_monthly_cost_usd,
            "status": "Severe CPU Bottleneck (+28ms/tok)",
        },
        {
            "tier_id": "tier3_xgrammar_llguidance_naive_schema",
            "name": "Tier 3: 预编译零开销约束解码 (XGrammar / llguidance，但字段顺序把结论排在推理前)",
            "json_syntax_valid_pct": t3_json_syntax_pass_rate_pct,
            "business_semantic_accuracy_pct": t3_business_semantic_accuracy_pct,
            "per_token_grammar_overhead_ms": t3_per_token_mask_overhead_ms,
            "p99_e2e_latency_ms": t3_p99_e2e_latency_ms,
            "monthly_cost_20m_calls_usd": t3_monthly_cost_usd,
            "status": "100% Valid Syntax, Suboptimal Semantics",
        },
        {
            "tier_id": "tier4_reasoning_first_schema_xgrammar_pydantic_gate",
            "name": "Tier 4: 推理字段前置契约 (Reasoning-First) + 扁平化 XGrammar + Pydantic 业务规则校验",
            "json_syntax_valid_pct": t4_json_syntax_pass_rate_pct,
            "business_semantic_accuracy_pct": t4_business_semantic_accuracy_pct,
            "per_token_grammar_overhead_ms": t4_per_token_mask_overhead_ms,
            "p99_e2e_latency_ms": t4_p99_e2e_latency_ms,
            "monthly_cost_20m_calls_usd": t4_monthly_cost_usd,
            "status": "Enterprise API Contract Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_structured_outputs_json_schema_constrained_decoding_selection",
        "round": 387,
        "title": "Enterprise Structured Outputs, JSON Schema Constrained Decoding & API Contract Architecture Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_xgrammar_llguidance_and_jsonschemabench_evaluations",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_api_calls": monthly_calls,
            "model_vocab_size": 128_256,
            "max_schema_nesting_depth": 3,
        },
        "headline_findings": {
            "tier4_json_syntax_valid_pct": t4_json_syntax_pass_rate_pct,
            "tier4_business_semantic_accuracy_pct": t4_business_semantic_accuracy_pct,
            "semantic_gain_t4_vs_t3_pts": semantic_gain_t4_vs_t3_pts,
            "latency_speedup_t4_vs_t1_x": latency_speedup_t4_vs_t1_x,
            "cost_savings_t4_vs_t1_pct": cost_savings_t4_vs_t1_pct,
            "tier4_per_token_overhead_ms": t4_per_token_mask_overhead_ms,
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
        print("OK: enterprise_structured_outputs_json_schema_constrained_decoding_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
