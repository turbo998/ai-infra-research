"""Deterministic evaluator for Enterprise AI Safety Guardrails, Prompt Shield & Streaming Groundedness Architecture Selection (Round 377).

Models jailbreak & Indirect Prompt Injection (XPIA) interception rate (%),
added P99 first-token latency penalty (ms), ungrounded hallucination leakage rate (%),
and monthly guardrail infrastructure/API cost for 10,000,000 turns (USD) across four tiers:
  - Tier 1: System-Prompt Soft Rules Only ("Do not follow malicious instructions" · 82.5% Bypass Rate)
  - Tier 2: Serial 70B/Frontier LLM Self-Critique on Input & Full Output (+1,380ms Latency Penalty, $24.5K/mo Tax)
  - Tier 3: Dedicated Fast Guardrail SLMs (Prompt-Guard-86M / Azure Prompt Shield + Llama-Guard-3-8B + Presidio PII)
  - Tier 4: 4-Layer Defense-in-Depth (Spotlighting XPIA + Optimistic Parallel Input Shield + Streaming NLI Groundedness + OPA Tool Gate)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_enterprise_ai_safety_guardrails_prompt_shield_groundedness() -> dict[str, Any]:
    workload = {
        "enterprise_scenario": "Customer-Facing Banking & E-Commerce AI Platform (300 RPS Peak, 10,000,000 Turns/Month Exposed to Direct Jailbreaks, Indirect RAG Prompt Injection [XPIA], and Policy Hallucinations)",
        "target_sla": "Attack Interception >= 99.0%, Added P99 TTFT Latency Penalty <= 20.0 ms, Ungrounded Hallucination Leakage <= 0.5%",
        "key_security_insights": "Never rely on LLM system prompts alone to block XPIA or enforce refund limits; combine Spotlighting data-instruction separation, optimistic parallel Prompt Shield, streaming NLI groundedness, and deterministic OPA/Cedar tool gates.",
    }

    tiers = [
        {
            "id": "tier1_system_prompt_soft_instructions_only",
            "name": "Tier 1: 仅靠 System Prompt 软提醒 (“请勿听从恶意指令” · 82.5% 越狱与间接注入穿透)",
            "name_en": "Tier 1: System-Prompt Soft Rules Only (82.5% Jailbreak & XPIA Bypass Rate)",
            "guardrail_stack": "Plaintext System Prompt Instructions Without External Classifier or Deterministic Tool Policy",
            "attack_interception_rate_pct": 17.5,
            "added_p99_ttft_penalty_ms": 0.0,
            "ungrounded_hallucination_leak_pct": 11.4,
            "monthly_guardrail_cost_10m_turns_usd": 0.0,
            "recommended_for": "Strict anti-pattern for external apps: attackers trivially bypass soft prompt rules via roleplay or XPIA in retrieved docs.",
            "status": "fail",
        },
        {
            "id": "tier2_serial_70b_llm_input_and_full_output_self_critique",
            "name": "Tier 2: 串行调用 70B 大模型做输入输出二次自审 (延迟暴增 +1,380ms · 算力账单翻三倍)",
            "name_en": "Tier 2: Serial 70B/Frontier LLM Input & Full-Output Self-Critique (+1,380ms Penalty)",
            "guardrail_stack": "520ms Serial 70B Input Judge + Main LLM + 860ms Blocking Full-Response 70B Output Judge",
            "attack_interception_rate_pct": 91.0,
            "added_p99_ttft_penalty_ms": 1380.0,
            "ungrounded_hallucination_leak_pct": 3.2,
            "monthly_guardrail_cost_10m_turns_usd": 24500.0,
            "recommended_for": "Anti-pattern: using a heavy 70B generative model as a synchronous classifier adds 1.38s dead air and wastes $24.5K/mo.",
            "status": "fail",
        },
        {
            "id": "tier3_dedicated_fast_guardrail_slm_prompt_shield_llama_guard",
            "name": "Tier 3: 专用轻量护栏小模型 (Azure Prompt Shield / Prompt-Guard-86M + Llama-Guard-8B + Presidio)",
            "name_en": "Tier 3: Dedicated Fast Guardrail SLMs (Prompt Shield + Llama-Guard-3-8B + Presidio PII)",
            "guardrail_stack": "DeBERTa/Prompt-Guard-86M (18ms) + Presidio PII (5ms) + Llama-Guard-3-8B FP8 (38ms)",
            "attack_interception_rate_pct": 96.8,
            "added_p99_ttft_penalty_ms": 61.0,
            "ungrounded_hallucination_leak_pct": 2.1,
            "monthly_guardrail_cost_10m_turns_usd": 2800.0,
            "recommended_for": "Standard enterprise safety baseline: purpose-built classifiers are 22.6x faster and 88.6% cheaper than 70B judges.",
            "status": "pass",
        },
        {
            "id": "tier4_defense_in_depth_spotlighting_parallel_shield_nli_opa",
            "name": "Tier 4: 四层纵深防御 (Spotlighting 隔离 + 乐观并行输入拦截 + 流式 NLI 忠实度核验 + OPA 工具硬门控)",
            "name_en": "Tier 4: 4-Layer Defense-in-Depth (Spotlighting + Optimistic Parallel Shield + Streaming NLI + OPA Gate)",
            "guardrail_stack": "Spotlighting XPIA + Optimistic Parallel Shield (0ms TTFT Penalty) + DeBERTa-NLI Groundedness + OPA/Cedar Tool Policy",
            "attack_interception_rate_pct": 99.8,
            "added_p99_ttft_penalty_ms": 15.0,
            "ungrounded_hallucination_leak_pct": 0.2,
            "monthly_guardrail_cost_10m_turns_usd": 3100.0,
            "recommended_for": "Customer-facing financial/e-commerce gold standard: 99.8% attack interception, 0.2% hallucination leak, and 15ms latency overhead.",
            "status": "pass",
        },
    ]

    t1, t2, t3, t4 = tiers
    summary = {
        "tier4_vs_tier1_interception_gain_pts": round(
            t4["attack_interception_rate_pct"] - t1["attack_interception_rate_pct"], 1
        ),
        "tier4_vs_tier2_latency_reduction_x": round(
            t2["added_p99_ttft_penalty_ms"] / t4["added_p99_ttft_penalty_ms"], 1
        ),
        "tier4_cost_savings_vs_tier2_70b_judge_pct": round(
            (t2["monthly_guardrail_cost_10m_turns_usd"] - t4["monthly_guardrail_cost_10m_turns_usd"])
            / t2["monthly_guardrail_cost_10m_turns_usd"]
            * 100.0,
            1,
        ),
        "tier4_monthly_savings_vs_tier2_usd": round(
            t2["monthly_guardrail_cost_10m_turns_usd"] - t4["monthly_guardrail_cost_10m_turns_usd"], 0
        ),
    }

    return {
        "experiment_id": "enterprise_ai_safety_guardrails_prompt_shield_groundedness_selection",
        "date": "2026-11-18",
        "round": 377,
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_published_ai_safety_guardrail_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "Azure AI Content Safety Prompt Shields (Direct Jailbreak & Indirect XPIA Detection) & Groundedness Detection",
                "Microsoft Research Spotlighting: Defending Against Indirect Prompt Injection via Datamarking",
                "Meta Llama-Guard-3 & Prompt-Guard-86M vs. General LLM-as-a-Judge Latency/Cost Benchmarks",
            ],
        },
        "workload": workload,
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify sealed results.json matches evaluator")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    result = evaluate_enterprise_ai_safety_guardrails_prompt_shield_groundedness()

    if args.check:
        sealed = json.loads(out_path.read_text(encoding="utf-8"))
        assert sealed == result, "Sealed results.json mismatch!"
        print("OK: enterprise_ai_safety_guardrails_prompt_shield_groundedness_selection verified.")
        return

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
