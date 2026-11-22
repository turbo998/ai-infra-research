from pathlib import Path
from experiments.enterprise_ai_solution_architect_20_formulas_10_redlines_master_handbook_selection.run import build_results


def test_enterprise_ai_solution_architect_20_formulas_10_redlines_master_handbook_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_ai_solution_architect_20_formulas_10_redlines_master_handbook_selection"
    assert len(res["ten_architectural_redlines"]) == 10
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["sizing_accuracy_vs_production_pct"] > tiers[0]["sizing_accuracy_vs_production_pct"]
    assert tiers[3]["effective_cost_per_1m_tokens_usd"] < tiers[0]["effective_cost_per_1m_tokens_usd"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-ai-solution-architect-20-formulas-10-redlines-master-handbook-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-ai-solution-architect-20-formulas-10-redlines-master-handbook-selection-faq.en.md").exists()
    assert (root / "docs/h5/milestone-465-sa-20-formulas-10-redlines-handbook-lab.html").exists()
