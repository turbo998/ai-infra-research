from pathlib import Path
from experiments.enterprise_rocev2_rail_optimized_pxn_dlb_packet_spraying_topology_sizing_selection.run import build_results


def test_enterprise_rocev2_rail_optimized_pxn_dlb_packet_spraying_topology_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_rocev2_rail_optimized_pxn_dlb_packet_spraying_topology_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["effective_busbw_gbps"] > tiers[1]["effective_busbw_gbps"] > tiers[0]["effective_busbw_gbps"]
    assert tiers[3]["p99_alltoall_tail_us"] < tiers[1]["p99_alltoall_tail_us"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-rocev2-rail-optimized-pxn-dlb-packet-spraying-topology-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-rocev2-rail-optimized-pxn-dlb-packet-spraying-topology-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/rocev2-rail-optimized-pxn-dlb-packet-spraying-lab.html").exists()
