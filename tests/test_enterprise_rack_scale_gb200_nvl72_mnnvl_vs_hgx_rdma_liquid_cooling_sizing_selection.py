from pathlib import Path
from experiments.enterprise_rack_scale_gb200_nvl72_mnnvl_vs_hgx_rdma_liquid_cooling_sizing_selection.run import build_results


def test_enterprise_rack_scale_gb200_nvl72_mnnvl_vs_hgx_rdma_liquid_cooling_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_rack_scale_gb200_nvl72_mnnvl_vs_hgx_rdma_liquid_cooling_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["p99_tpot_ms"] < tiers[1]["p99_tpot_ms"] < tiers[0]["p99_tpot_ms"]
    assert tiers[3]["per_gpu_goodput_tok_s"] > tiers[1]["per_gpu_goodput_tok_s"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-rack-scale-gb200-nvl72-mnnvl-vs-hgx-rdma-liquid-cooling-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-rack-scale-gb200-nvl72-mnnvl-vs-hgx-rdma-liquid-cooling-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/rack-scale-gb200-nvl72-mnnvl-vs-hgx-rdma-tco-lab.html").exists()
