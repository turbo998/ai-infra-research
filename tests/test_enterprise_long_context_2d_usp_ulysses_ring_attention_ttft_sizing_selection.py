from pathlib import Path
from experiments.enterprise_long_context_2d_usp_ulysses_ring_attention_ttft_sizing_selection.run import build_results


def test_enterprise_long_context_2d_usp_ulysses_ring_attention_ttft_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_long_context_2d_usp_ulysses_ring_attention_ttft_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["prefill_256k_ttft_sec"] < tiers[2]["prefill_256k_ttft_sec"] < tiers[0]["prefill_256k_ttft_sec"]
    assert tiers[3]["max_supported_context_len_k"] >= 1024

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-long-context-2d-usp-ulysses-ring-attention-ttft-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-long-context-2d-usp-ulysses-ring-attention-ttft-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/long-context-2d-usp-ulysses-ring-ttft-sizer-lab.html").exists()
