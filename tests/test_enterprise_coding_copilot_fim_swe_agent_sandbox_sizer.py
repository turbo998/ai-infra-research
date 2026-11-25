from experiments.enterprise_coding_copilot_fim_swe_agent_sandbox_sizer.run import size_coding_platform


def test_enterprise_coding_copilot_fim_swe_agent_sandbox_sizer() -> None:
    res = size_coding_platform(10000, 0.62)
    assert res["Deliverable_ID"] == "#644"
    assert res["Track_A_FIM"]["effective_rps"] == 149.0
    assert res["Track_A_FIM"]["l40s_gpus"] == 16
    assert res["Track_B_SWE_Agent"]["h200_gpus"] == 16
    assert res["Track_B_SWE_Agent"]["ast_prefix_cache_hit_pct"] == 84.2
    assert res["MicroVM_Sandbox_Pool"]["concurrent_microvms"] == 240
    assert res["FinOps_TCO"]["tco_saving_pct"] == 79.8
