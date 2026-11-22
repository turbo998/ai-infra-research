from pathlib import Path
from experiments.enterprise_multi_silicon_gpu_tpu_h200_b200_mi325x_tpu_v6e_tco_sizing_selection.run import build_results


def test_enterprise_multi_silicon_gpu_tpu_h200_b200_mi325x_tpu_v6e_tco_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_multi_silicon_gpu_tpu_h200_b200_mi325x_tpu_v6e_tco_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["cost_per_1m_goodput_tokens_usd"] < tiers[1]["cost_per_1m_goodput_tokens_usd"] < tiers[0]["cost_per_1m_goodput_tokens_usd"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-multi-silicon-gpu-tpu-h200-b200-mi325x-tpu-v6e-tco-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-multi-silicon-gpu-tpu-h200-b200-mi325x-tpu-v6e-tco-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/multi-silicon-h200-b200-mi325x-tpu-v6e-tco-sizer-lab.html").exists()
