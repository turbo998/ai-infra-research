from experiments.enterprise_gpu_sizer_three_watermarks_tpm_tps_littles_law_sla_sizing.run import build_summary


def test_enterprise_gpu_sizer_three_watermarks_tpm_tps_littles_law_sla_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["three_watermark_pareto_goodput_pd_mtp_12gpu"]
    wb = s["workload_baseline"]
    assert wb["littles_law_b_active"] == 242
    assert opt["required_h200_gpus"] == max(
        opt["watermark_1_mem_gpus"],
        opt["watermark_2_bw_gpus"],
        opt["watermark_3_compute_gpus"],
    )
    assert opt["sla_goodput_pct"] == 99.2
    assert opt["p99_tpot_ms"] == 13.2
