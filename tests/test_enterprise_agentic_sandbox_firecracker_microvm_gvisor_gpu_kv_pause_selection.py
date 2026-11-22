from pathlib import Path
from experiments.enterprise_agentic_sandbox_firecracker_microvm_gvisor_gpu_kv_pause_selection.run import build_results


def test_enterprise_agentic_sandbox_firecracker_microvm_gvisor_gpu_kv_pause_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_agentic_sandbox_firecracker_microvm_gvisor_gpu_kv_pause_selection"
    rts = res["sandbox_runtimes"]
    assert len(rts) == 4
    assert rts[3]["cold_spawn_or_restore_ms"] < rts[0]["cold_spawn_or_restore_ms"]
    assert rts[3]["gpu_hbm_wasted_during_tool_wait_pct"] < 10.0
    assert rts[3]["effective_agent_trajectories_per_hour_8gpu"] > rts[0]["effective_agent_trajectories_per_hour_8gpu"] * 3

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-agentic-sandbox-firecracker-microvm-gvisor-gpu-kv-pause-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-agentic-sandbox-firecracker-microvm-gvisor-gpu-kv-pause-selection-faq.en.md").exists()
    assert (root / "docs/h5/agentic-sandbox-firecracker-microvm-gpu-kv-pause-lab.html").exists()
