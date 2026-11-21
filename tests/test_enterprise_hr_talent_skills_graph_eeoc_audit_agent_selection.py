from pathlib import Path
from experiments.enterprise_hr_talent_skills_graph_eeoc_audit_agent_selection.run import build_results


def test_enterprise_hr_talent_skills_graph_eeoc_audit_agent_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_hr_talent_skills_graph_eeoc_audit_agent_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["eeoc_adverse_impact_ratio"] >= 0.80
    assert tiers[3]["qualified_candidate_recall_pct"] > tiers[0]["qualified_candidate_recall_pct"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-hr-talent-skills-graph-eeoc-audit-agent-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-hr-talent-skills-graph-eeoc-audit-agent-selection-faq.en.md").exists()
    assert (root / "docs/h5/hr-talent-skills-graph-eeoc-audit-agent-lab.html").exists()
