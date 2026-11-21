from pathlib import Path
from experiments.enterprise_llm_evaluation_judge_promptfoo_ragas_shadow_canary_tco_selection.run import build_results


def test_enterprise_llm_evaluation_judge_promptfoo_ragas_shadow_canary_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_llm_evaluation_judge_promptfoo_ragas_shadow_canary_tco_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["human_agreement_kappa"] > tiers[1]["human_agreement_kappa"]
    assert tiers[3]["monthly_eval_tco_usd"] < tiers[1]["monthly_eval_tco_usd"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-llm-evaluation-judge-promptfoo-ragas-shadow-canary-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-llm-evaluation-judge-promptfoo-ragas-shadow-canary-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/llm-evaluation-judge-promptfoo-ragas-shadow-canary-lab.html").exists()
