from experiments.enterprise_zero_ttft_ai_guardrail_speculative_sidecar_pii.run import (
    evaluate_guardrail_architecture,
)


def test_enterprise_zero_ttft_ai_guardrail_speculative_sidecar_pii() -> None:
    res = evaluate_guardrail_architecture()
    assert res["Deliverable_ID"] == "#647"
    lat = res["Latency_Comparison"]
    assert lat["net_delta_ms"] == 2.5
    assert lat["delta_reduction_pct"] == 98.3
    assert res["Output_Streaming_Guardrail"]["sliding_window_tokens"] == 6
    assert res["Sidecar_Sizing_100RPS"]["sidecar_l40s_gpus"] == 2
    assert res["Sidecar_Sizing_100RPS"]["cost_saving_vs_cloud_api_pct"] == 90.4
