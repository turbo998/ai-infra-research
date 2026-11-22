from pathlib import Path
from experiments.enterprise_moe_expert_load_balancing_eplb_deepep_ibgda_sizing_selection.run import build_results


def test_enterprise_moe_expert_load_balancing_eplb_deepep_ibgda_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_moe_expert_load_balancing_eplb_deepep_ibgda_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["p99_tpot_ms"] < tiers[1]["p99_tpot_ms"] < tiers[0]["p99_tpot_ms"]
    assert tiers[3]["per_gpu_goodput_tok_s"] > tiers[0]["per_gpu_goodput_tok_s"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-moe-expert-load-balancing-eplb-deepep-ibgda-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-moe-expert-load-balancing-eplb-deepep-ibgda-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/moe-expert-load-balancing-eplb-deepep-ibgda-lab.html").exists()
