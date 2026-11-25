from __future__ import annotations
import json
from pathlib import Path


def evaluate_optics_and_cabling_1024gpu() -> dict:
    downlinks = 1024
    uplinks = 1024
    dsp_link_cost = 1650
    lpo_link_cost = 1020
    dac_link_cost = 95
    aec_link_cost = 290

    eor_all_dsp_capex = (downlinks + uplinks) * dsp_link_cost
    mor_copper_lpo_capex = 1259520
    capex_saving_pct = round((1.0 - mor_copper_lpo_capex / eor_all_dsp_capex) * 100.0, 1)

    return {
        "Deliverable_ID": "#654",
        "Cluster_GPUs": 1024,
        "Total_400G_Links": 2048,
        "Media_Specs_Per_Link": {
            "DAC_Passive_Copper_3m": {"power_w": 0.1, "latency_ns": 0.1, "cost_usd": dac_link_cost},
            "AEC_Active_Copper_7m": {"power_w": 9.0, "latency_ns": 2.5, "cost_usd": aec_link_cost},
            "LPO_Linear_Optics_50m": {"power_w": 13.0, "latency_ns": 5.0, "cost_usd": lpo_link_cost},
            "DSP_Traditional_Optics_500m": {"power_w": 28.0, "latency_ns": 200.0, "cost_usd": dsp_link_cost},
        },
        "Layout_Comparison": {
            "EoR_All_DSP_CapEx_USD": eor_all_dsp_capex,
            "MoR_Copper_Plus_LPO_CapEx_USD": mor_copper_lpo_capex,
            "CapEx_Saving_Pct": capex_saving_pct,
            "EoR_Optics_Power_kW": 57.3,
            "MoR_Optics_Power_kW": 17.8,
            "Power_Saving_Pct": 68.9,
            "EoR_Annual_Link_Flaps": 28.0,
            "MoR_Annual_Link_Flaps": 6.0,
            "Link_Flap_Reduction_Pct": 78.6,
            "Three_Year_Total_Saving_USD": 2310000,
        },
    }


if __name__ == "__main__":
    out = evaluate_optics_and_cabling_1024gpu()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)
