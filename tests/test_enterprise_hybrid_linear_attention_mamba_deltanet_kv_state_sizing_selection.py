from pathlib import Path
from experiments.enterprise_hybrid_linear_attention_mamba_deltanet_kv_state_sizing_selection.run import build_results


def test_enterprise_hybrid_linear_attention_mamba_deltanet_kv_state_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_hybrid_linear_attention_mamba_deltanet_kv_state_sizing_selection"
    archs = res["attention_architectures"]
    assert len(archs) == 4
    assert archs[3]["kv_and_state_per_256k_req_gb"] < archs[0]["kv_and_state_per_256k_req_gb"] * 0.15
    assert archs[3]["needle_in_haystack_recall_pct"] > 98.5
    assert archs[3]["max_concurrent_256k_sessions_8gpu"] > archs[0]["max_concurrent_256k_sessions_8gpu"] * 7

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-hybrid-linear-attention-mamba-deltanet-kv-state-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-hybrid-linear-attention-mamba-deltanet-kv-state-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/hybrid-linear-attention-mamba-deltanet-kv-sizer-lab.html").exists()
