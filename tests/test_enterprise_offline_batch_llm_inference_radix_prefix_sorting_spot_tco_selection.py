from pathlib import Path
from experiments.enterprise_offline_batch_llm_inference_radix_prefix_sorting_spot_tco_selection.run import build_results


def test_enterprise_offline_batch_llm_inference_radix_prefix_sorting_spot_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_offline_batch_llm_inference_radix_prefix_sorting_spot_tco_selection"
    pipes = res["batch_pipelines"]
    assert len(pipes) == 4
    assert pipes[3]["prefix_cache_hit_rate_pct"] > 90.0
    assert pipes[3]["throughput_tokens_per_sec_per_gpu"] > pipes[0]["throughput_tokens_per_sec_per_gpu"] * 6
    assert pipes[3]["effective_cost_per_1m_tokens_usd"] < 0.035

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-offline-batch-llm-inference-radix-prefix-sorting-spot-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-offline-batch-llm-inference-radix-prefix-sorting-spot-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/offline-batch-llm-radix-sorting-spot-tco-lab.html").exists()
