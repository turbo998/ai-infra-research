from __future__ import annotations

from experiments.enterprise_ai_datacenter_power_transient_bess_dlc_retrofit_sizer.run import (
    simulate_ai_datacenter_power_and_dlc,
)


def test_ai_datacenter_power_transient_and_dlc_sizer() -> None:
    res = simulate_ai_datacenter_power_and_dlc()
    tm = res["transient_mitigation"]
    assert tm["unmitigated_raw_gpu_step_load"]["step_swing_pct_in_20ms"] == 75.0
    assert tm["three_layer_power_smoothing_cbu_bess_burn_in"]["step_swing_pct_in_20ms"] == 11.5
    assert tm["three_layer_power_smoothing_cbu_bess_burn_in"]["workload_throughput_penalty_pct"] == 0.8
    cr = res["cooling_retrofits"]
    assert cr["liquid_to_air_sidecar_cdu_l2a"]["max_kw_per_rack"] == 75
    assert cr["liquid_to_liquid_in_row_cdu_l2l"]["design_pue"] == 1.11
