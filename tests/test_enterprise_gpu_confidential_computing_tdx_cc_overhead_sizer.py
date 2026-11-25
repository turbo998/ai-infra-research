from __future__ import annotations

from experiments.enterprise_gpu_confidential_computing_tdx_cc_overhead_sizer.run import (
    simulate_confidential_computing_overhead,
)


def test_gpu_confidential_computing_overhead_sizer() -> None:
    res = simulate_confidential_computing_overhead()
    modes = res["modes"]
    assert modes["naive_cc_on_pcie_bounce_bf16_kv"]["tp8_70b_pci_bounce_tps_overhead_pct"] == 28.5
    assert modes["optimized_ppc_ie_nvlink_cc_fp8_kv"]["tp8_70b_nvlink_ppc_ie_tps_overhead_pct"] == 4.8
    assert modes["optimized_ppc_ie_nvlink_cc_fp8_kv"]["single_gpu_8b_tps_overhead_pct"] <= 2.1
    assert res["banking_fleet_sizing"]["gpu_savings_ppcie_vs_naive_pct"] == 26.1
