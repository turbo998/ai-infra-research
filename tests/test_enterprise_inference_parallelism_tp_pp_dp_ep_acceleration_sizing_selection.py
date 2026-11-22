from pathlib import Path
from experiments.enterprise_inference_parallelism_tp_pp_dp_ep_acceleration_sizing_selection.run import build_results


def test_enterprise_inference_parallelism_tp_pp_dp_ep_acceleration_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_inference_parallelism_tp_pp_dp_ep_acceleration_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["p99_tpot_ms"] < tiers[2]["p99_tpot_ms"] < tiers[0]["p99_tpot_ms"]
    assert tiers[3]["per_gpu_goodput_tok_s"] > tiers[1]["per_gpu_goodput_tok_s"] > tiers[0]["per_gpu_goodput_tok_s"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-inference-parallelism-tp-pp-dp-ep-acceleration-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-inference-parallelism-tp-pp-dp-ep-acceleration-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/inference-parallelism-tp-pp-dp-ep-acceleration-matrix-lab.html").exists()
