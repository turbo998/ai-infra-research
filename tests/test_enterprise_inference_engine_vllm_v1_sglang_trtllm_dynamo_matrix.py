from experiments.enterprise_inference_engine_vllm_v1_sglang_trtllm_dynamo_matrix.run import (
    evaluate_inference_engine_matrix,
)


def test_enterprise_inference_engine_vllm_v1_sglang_trtllm_dynamo_matrix() -> None:
    res = evaluate_inference_engine_matrix()
    assert res["Deliverable_ID"] == "#656"
    assert res["Exposed_CPU_Overhead_ms"]["cpu_overhead_reduction_pct"] == 97.8
    wk = res["Workloads_8xH200"]
    assert wk["Single_Turn_70B_B64_TPS"]["v1_speedup_over_v0"] == 1.52
    assert wk["Multi_Agent_JSON_B64_TPS"]["sglang"] == 15800
    assert wk["Low_Latency_B8_TPOT_ms"]["trt_llm"] == 4.9
