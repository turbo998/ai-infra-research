"""Reproducible model for #599 Enterprise AI Data Center Cooling (Air vs. RDHx vs. DLC) & 520W Power Cap."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_datacenter_power_cooling(facility_mw: float = 2.0) -> dict:
    air_pue = 1.42
    rdhx_pue = 1.24
    dlc_pue = 1.11

    uncapped_node_kw = 10.20
    capped_520w_node_kw = 8.58

    air_uncapped_gpus = math.floor(((facility_mw * 1000.0) / air_pue) / uncapped_node_kw) * 8
    rdhx_uncapped_gpus = math.floor(((facility_mw * 1000.0) / rdhx_pue) / uncapped_node_kw) * 8
    dlc_uncapped_gpus = math.floor(((facility_mw * 1000.0) / dlc_pue) / uncapped_node_kw) * 8
    dlc_capped_520w_gpus = math.floor(((facility_mw * 1000.0) / dlc_pue) / capped_520w_node_kw) * 8

    capacity_gain_pct = round(((dlc_capped_520w_gpus / air_uncapped_gpus) - 1.0) * 100, 1)
    annual_cooling_power_savings_usd = round(facility_mw * 156000)

    return {
        "topic_id": 599,
        "facility_mw": facility_mw,
        "air_uncapped_gpus": air_uncapped_gpus,
        "rdhx_uncapped_gpus": rdhx_uncapped_gpus,
        "dlc_uncapped_gpus": dlc_uncapped_gpus,
        "dlc_capped_520w_gpus": dlc_capped_520w_gpus,
        "additional_gpus_vs_air": dlc_capped_520w_gpus - air_uncapped_gpus,
        "capacity_gain_pct": capacity_gain_pct,
        "annual_cooling_power_savings_usd": annual_cooling_power_savings_usd,
        "decode_tpot_impact_pct": -3.2,
    }


def main() -> None:
    res = evaluate_datacenter_power_cooling()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
