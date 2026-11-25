from experiments.enterprise_ai_cluster_optics_dac_aec_lpo_vs_dsp_mor_tco.run import (
    evaluate_optics_and_cabling_1024gpu,
)


def test_enterprise_ai_cluster_optics_dac_aec_lpo_vs_dsp_mor_tco() -> None:
    res = evaluate_optics_and_cabling_1024gpu()
    assert res["Deliverable_ID"] == "#654"
    cmp = res["Layout_Comparison"]
    assert cmp["CapEx_Saving_Pct"] == 62.7
    assert cmp["Power_Saving_Pct"] == 68.9
    assert cmp["Link_Flap_Reduction_Pct"] == 78.6
    assert cmp["Three_Year_Total_Saving_USD"] == 2310000
