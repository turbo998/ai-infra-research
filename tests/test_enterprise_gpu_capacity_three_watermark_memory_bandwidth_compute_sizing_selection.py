from pathlib import Path
from experiments.enterprise_gpu_capacity_three_watermark_memory_bandwidth_compute_sizing_selection.run import build_results


def test_enterprise_gpu_capacity_three_watermark_memory_bandwidth_compute_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_gpu_capacity_three_watermark_memory_bandwidth_compute_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["p99_tpot_ms"] < tiers[2]["p99_tpot_ms"] < tiers[1]["p99_tpot_ms"] < tiers[0]["p99_tpot_ms"]
    assert tiers[3]["cost_per_1m_goodput_tokens_usd"] < tiers[0]["cost_per_1m_goodput_tokens_usd"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-gpu-capacity-three-watermark-memory-bandwidth-compute-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-gpu-capacity-three-watermark-memory-bandwidth-compute-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/gpu-three-watermark-memory-bandwidth-compute-sizer-lab.html").exists()
