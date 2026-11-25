from experiments.enterprise_llm_quantization_bf16_fp8_int4_awq_nvfp4_kv_sizer.run import (
    evaluate_quantization_matrix,
)


def test_enterprise_llm_quantization_bf16_fp8_int4_awq_nvfp4_kv_sizer() -> None:
    res = evaluate_quantization_matrix()
    assert res["Deliverable_ID"] == "#651"
    assert res["Crossover_Batch_Size"] == 24
    s = res["Schemes_70B_4GPU_B64"]
    assert s["Block_FP8_W8A8KV8"]["speedup_vs_w4a16_at_b64"] == 1.48
    assert s["Block_FP8_W8A8KV8"]["speedup_vs_bf16"] == 2.28
    assert s["Blackwell_Hybrid_FP8_NVFP4_KV8"]["speedup_vs_bf16"] == 3.85
