from pathlib import Path
from experiments.enterprise_devsecops_sast_copilot_autofix_codeql_cpg_agent_tco_selection.run import build_results


def test_enterprise_devsecops_sast_copilot_autofix_codeql_cpg_agent_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_devsecops_sast_copilot_autofix_codeql_cpg_agent_tco_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["false_positive_rate_pct"] < tiers[0]["false_positive_rate_pct"]
    assert tiers[3]["patch_accept_compile_pass_pct"] > tiers[1]["patch_accept_compile_pass_pct"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-devsecops-sast-copilot-autofix-codeql-cpg-agent-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-devsecops-sast-copilot-autofix-codeql-cpg-agent-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/devsecops-sast-copilot-autofix-codeql-cpg-agent-lab.html").exists()
