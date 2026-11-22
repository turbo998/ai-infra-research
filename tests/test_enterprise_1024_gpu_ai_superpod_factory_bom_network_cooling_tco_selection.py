from pathlib import Path
from experiments.enterprise_1024_gpu_ai_superpod_factory_bom_network_cooling_tco_selection.run import build_results


def test_enterprise_1024_gpu_ai_superpod_factory_bom_network_cooling_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_1024_gpu_ai_superpod_factory_bom_network_cooling_tco_selection"
    assert res["total_gpus"] == 1024
    designs = res["superpod_designs"]
    assert len(designs) == 4
    assert designs[3]["three_year_total_tco_million_usd"] < designs[0]["three_year_total_tco_million_usd"]
    assert designs[3]["effective_cluster_goodput_utilization_pct"] > 90.0
    assert designs[3]["effective_cost_per_1m_tokens_usd"] < 0.10

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-1024-gpu-ai-superpod-factory-bom-network-cooling-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-1024-gpu-ai-superpod-factory-bom-network-cooling-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/milestone-475-1024-gpu-superpod-bom-tco-lab.html").exists()
