from pathlib import Path
from experiments.enterprise_tiered_kv_cache_offload_dram_nvme_gds_recompute_sizing_selection import run


def test_enterprise_tiered_kv_cache_offload_dram_nvme_gds_recompute_sizing_selection() -> None:
    res = run.build_results()
    assert res["experiment_id"] == "enterprise_tiered_kv_cache_offload_dram_nvme_gds_recompute_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["resumed_turn_ttft_ms"] < tiers[2]["resumed_turn_ttft_ms"] < tiers[0]["resumed_turn_ttft_ms"]
    assert tiers[3]["effective_cache_hit_pct"] > tiers[0]["effective_cache_hit_pct"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-tiered-kv-cache-offload-dram-nvme-gds-recompute-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-tiered-kv-cache-offload-dram-nvme-gds-recompute-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/tiered-kv-cache-offload-dram-nvme-gds-recompute-lab.html").exists()
