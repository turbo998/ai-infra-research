from pathlib import Path
from experiments.enterprise_onprem_edge_ai_appliance_rtx_pro_6000_l40s_h200_nvl_selection.run import build_results


def test_enterprise_onprem_edge_ai_appliance_rtx_pro_6000_l40s_h200_nvl_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_onprem_edge_ai_appliance_rtx_pro_6000_l40s_h200_nvl_selection"
    apps = res["appliances"]
    assert len(apps) == 4
    assert apps[1]["appliance_capex_usd"] < apps[0]["appliance_capex_usd"]
    assert apps[1]["llama_70b_fp8_single_user_tpot_ms"] < apps[0]["llama_70b_fp8_single_user_tpot_ms"]
    assert all(a["wall_power_watts"] <= res["standard_branch_rack_power_watts"] if "standard_branch_rack_power_watts" in res else a["wall_power_watts"] <= res["standard_branch_rack_power_limit_watts"] for a in apps)

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-onprem-edge-ai-appliance-rtx-pro-6000-l40s-h200-nvl-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-onprem-edge-ai-appliance-rtx-pro-6000-l40s-h200-nvl-selection-faq.en.md").exists()
    assert (root / "docs/h5/onprem-ai-appliance-rtx-pro-6000-l40s-h200-nvl-lab.html").exists()
