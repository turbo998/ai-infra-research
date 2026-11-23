from experiments.enterprise_lakehouse_batch_ai_etl_ray_data_vllm_spark_sizing.run import build_summary


def test_enterprise_lakehouse_batch_ai_etl_ray_data_vllm_spark_sizing() -> None:
    s = build_summary()
    assert s["throughput_speedup_vs_spark_udf"] == 13.44
    assert s["gpu_reduction_pct_vs_spark_udf"] == 87.5
    assert s["daily_cost_reduction_pct_vs_spark_udf"] == 94.3
    assert s["schemes"]["ray_data_streaming_inproc_vllm_fp8"]["required_h200_gpus"] == 16
    assert s["schemes"]["ray_data_streaming_inproc_vllm_fp8"]["cost_per_1k_docs_usd"] == 0.0284
