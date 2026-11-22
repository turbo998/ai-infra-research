from pathlib import Path
from experiments.enterprise_ai_optical_interconnect_dac_dsp_lpo_cpo_link_flap_tco_selection.run import build_results


def test_enterprise_ai_optical_interconnect_dac_dsp_lpo_cpo_link_flap_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_ai_optical_interconnect_dac_dsp_lpo_cpo_link_flap_tco_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["interconnect_3y_tco_1k_gpu_usd"] < tiers[1]["interconnect_3y_tco_1k_gpu_usd"] < tiers[0]["interconnect_3y_tco_1k_gpu_usd"]
    assert tiers[3]["annual_link_flap_incidents_1k_gpu"] < tiers[0]["annual_link_flap_incidents_1k_gpu"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-ai-optical-interconnect-dac-dsp-lpo-cpo-link-flap-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-ai-optical-interconnect-dac-dsp-lpo-cpo-link-flap-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/ai-optical-interconnect-dac-dsp-lpo-cpo-tco-lab.html").exists()
