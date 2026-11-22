from pathlib import Path
from experiments.enterprise_prompt_compression_kv_eviction_llmlingua_snapkv_radix_selection.run import build_results


def test_enterprise_prompt_compression_kv_eviction_llmlingua_snapkv_radix_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_prompt_compression_kv_eviction_llmlingua_snapkv_radix_selection"
    modes = res["compression_modes"]
    assert len(modes) == 4
    assert modes[3]["effective_prefill_ttft_ms"] < modes[1]["effective_prefill_ttft_ms"]
    assert modes[3]["prefix_cache_hit_rate_pct"] > 90.0
    assert modes[3]["exact_entity_code_accuracy_pct"] > 98.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-prompt-compression-kv-eviction-llmlingua-snapkv-radix-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-prompt-compression-kv-eviction-llmlingua-snapkv-radix-selection-faq.en.md").exists()
    assert (root / "docs/h5/prompt-compression-kv-eviction-llmlingua-radix-lab.html").exists()
