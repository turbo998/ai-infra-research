from pathlib import Path
from experiments.enterprise_vertical_industry_ai_infra_sizing_sla_tco_master_matrix_selection.run import build_results


def test_enterprise_vertical_industry_ai_infra_sizing_sla_tco_master_matrix_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_vertical_industry_ai_infra_sizing_sla_tco_master_matrix_selection"
    assert res["industry_count"] == 6
    inds = res["industry_blueprints"]
    assert len(inds) == 6
    assert all(i["tco_savings_vs_naive_arch_pct"] > 55.0 for i in inds)

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-vertical-industry-ai-infra-sizing-sla-tco-master-matrix-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-vertical-industry-ai-infra-sizing-sla-tco-master-matrix-selection-faq.en.md").exists()
    assert (root / "docs/h5/milestone-480-vertical-industry-ai-infra-matrix-lab.html").exists()
