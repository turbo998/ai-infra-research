from pathlib import Path
from experiments.enterprise_ai_solution_architect_cxo_defense_architecture_review_master_playbook_selection.run import build_results


def test_enterprise_ai_solution_architect_cxo_defense_architecture_review_master_playbook_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_ai_solution_architect_cxo_defense_architecture_review_master_playbook_selection"
    assert res["total_executive_qa_challenges"] == 50
    domains = res["review_domains"]
    assert len(domains) == 4
    assert sum(d["executive_challenge_count"] for d in domains) == 50

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-ai-solution-architect-cxo-defense-architecture-review-master-playbook-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-ai-solution-architect-cxo-defense-architecture-review-master-playbook-selection-faq.en.md").exists()
    assert (root / "docs/h5/milestone-490-sa-cxo-defense-architecture-review-lab.html").exists()
