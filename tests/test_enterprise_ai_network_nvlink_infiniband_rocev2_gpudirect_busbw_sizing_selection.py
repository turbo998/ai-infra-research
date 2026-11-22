from pathlib import Path
from experiments.enterprise_ai_network_nvlink_infiniband_rocev2_gpudirect_busbw_sizing_selection.run import build_results


def test_enterprise_ai_network_nvlink_infiniband_rocev2_gpudirect_busbw_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_ai_network_nvlink_infiniband_rocev2_gpudirect_busbw_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[2]["nccl_allreduce_busbw_gbps"] > tiers[1]["nccl_allreduce_busbw_gbps"] > tiers[0]["nccl_allreduce_busbw_gbps"]
    assert tiers[3]["p99_alltoall_latency_us"] < tiers[2]["p99_alltoall_latency_us"] < tiers[1]["p99_alltoall_latency_us"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-ai-network-nvlink-infiniband-rocev2-gpudirect-busbw-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-ai-network-nvlink-infiniband-rocev2-gpudirect-busbw-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/ai-network-nvlink-infiniband-rocev2-gpudirect-busbw-lab.html").exists()
