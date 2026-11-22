from pathlib import Path
from experiments.enterprise_multi_lora_serving_slora_punica_hbm_cache_tco_selection.run import build_results


def test_enterprise_multi_lora_serving_slora_punica_hbm_cache_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_multi_lora_serving_slora_punica_hbm_cache_tco_selection"
    modes = res["serving_modes"]
    assert len(modes) == 4
    assert modes[3]["gpus_required_128_tenants"] < modes[0]["gpus_required_128_tenants"] * 0.05
    assert modes[3]["mixed_batch_tpot_p99_ms"] < modes[1]["mixed_batch_tpot_p99_ms"]
    assert modes[3]["monthly_cluster_cost_usd"] < modes[0]["monthly_cluster_cost_usd"] * 0.05

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-multi-lora-serving-slora-punica-hbm-cache-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-multi-lora-serving-slora-punica-hbm-cache-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/multi-lora-serving-slora-punica-hbm-tco-lab.html").exists()
