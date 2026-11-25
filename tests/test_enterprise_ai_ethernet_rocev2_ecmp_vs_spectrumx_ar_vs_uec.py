from experiments.enterprise_ai_ethernet_rocev2_ecmp_vs_spectrumx_ar_vs_uec.run import (
    evaluate_ai_ethernet_generations,
)


def test_enterprise_ai_ethernet_rocev2_ecmp_vs_spectrumx_ar_vs_uec() -> None:
    res = evaluate_ai_ethernet_generations(400.0)
    assert res["Deliverable_ID"] == "#649"
    g = res["Generations"]
    assert g["Gen1_Classic_ECMP_RoCEv2"]["efficiency_pct"] == 53.8
    assert g["Gen1_Plus_Tuned_MultiQP4_OOO"]["busbw_gbs"] == 318.0
    assert g["Gen2_NVIDIA_SpectrumX_AR_DDP"]["efficiency_pct"] == 95.5
    assert g["Gen3_Ultra_Ethernet_UEC_Spraying"]["efficiency_pct"] == 94.0
