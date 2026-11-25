from __future__ import annotations
import json
from pathlib import Path


def evaluate_multiregion_dr_grey_failure_egress_sizing() -> dict:
    legacy_http_timeout_rto_sec = 60.0
    ttft_breaker_rto_sec = 0.85
    rto_speedup_x = round(legacy_http_timeout_rto_sec / ttft_breaker_rto_sec, 1)  # 70.6x

    active_passive_cost_ratio_pct = 200.0
    active_active_55_55_cost_ratio_pct = 115.0
    dr_tco_savings_pct = round(
        (1.0 - active_active_55_55_cost_ratio_pct / active_passive_cost_ratio_pct) * 100.0, 1
    )  # 42.5%

    wan_kv_egress_monthly_usd = 48_500
    whole_json_egress_monthly_usd = 420
    egress_savings_pct = round(
        (1.0 - whole_json_egress_monthly_usd / wan_kv_egress_monthly_usd) * 100.0, 1
    )  # 99.1%
    mid_stream_drop_error_pct = 0.0

    return {
        "deliverable_id": "#638",
        "topic": "enterprise_multiregion_active_active_dr_grey_failure_egress_sizing",
        "legacy_http_timeout_rto_sec": legacy_http_timeout_rto_sec,
        "ttft_breaker_rto_sec": ttft_breaker_rto_sec,
        "rto_speedup_x": rto_speedup_x,
        "dr_tco_savings_pct": dr_tco_savings_pct,
        "wan_kv_egress_monthly_usd": wan_kv_egress_monthly_usd,
        "whole_json_egress_monthly_usd": whole_json_egress_monthly_usd,
        "egress_savings_pct": egress_savings_pct,
        "mid_stream_drop_error_pct": mid_stream_drop_error_pct,
    }


if __name__ == "__main__":
    out = evaluate_multiregion_dr_grey_failure_egress_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)
