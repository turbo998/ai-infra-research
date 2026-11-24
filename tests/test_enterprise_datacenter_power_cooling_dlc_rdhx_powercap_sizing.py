from experiments.enterprise_datacenter_power_cooling_dlc_rdhx_powercap_sizing.run import evaluate_datacenter_power_cooling


def test_evaluate_datacenter_power_cooling():
    r = evaluate_datacenter_power_cooling(2.0)
    assert r["air_uncapped_gpus"] == 1104
    assert r["dlc_capped_520w_gpus"] == 1680
    assert r["additional_gpus_vs_air"] == 576
    assert r["capacity_gain_pct"] == 52.2
    assert r["annual_cooling_power_savings_usd"] == 312000
