from pathlib import Path
from experiments.enterprise_ai_safety_guardrails_speculative_parallel_shield_ttft_selection.run import build_results


def test_enterprise_ai_safety_guardrails_speculative_parallel_shield_ttft_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_ai_safety_guardrails_speculative_parallel_shield_ttft_selection"
    archs = res["guardrail_architectures"]
    assert len(archs) == 4
    assert archs[3]["added_ttft_p99_ms"] < 2.0
    assert archs[3]["jailbreak_prompt_injection_recall_pct"] > 97.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-ai-safety-guardrails-speculative-parallel-shield-ttft-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-ai-safety-guardrails-speculative-parallel-shield-ttft-selection-faq.en.md").exists()
    assert (root / "docs/h5/ai-safety-guardrails-speculative-parallel-ttft-lab.html").exists()
