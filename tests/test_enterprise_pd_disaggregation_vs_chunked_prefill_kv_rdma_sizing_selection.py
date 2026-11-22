from pathlib import Path
from experiments.enterprise_pd_disaggregation_vs_chunked_prefill_kv_rdma_sizing_selection.run import build_results


def test_enterprise_pd_disaggregation_vs_chunked_prefill_kv_rdma_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_pd_disaggregation_vs_chunked_prefill_kv_rdma_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["p99_tpot_ms"] < tiers[1]["p99_tpot_ms"] < tiers[0]["p99_tpot_ms"]
    assert tiers[3]["goodput_tok_s"] > tiers[1]["goodput_tok_s"] > tiers[0]["goodput_tok_s"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-pd-disaggregation-vs-chunked-prefill-kv-rdma-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-pd-disaggregation-vs-chunked-prefill-kv-rdma-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/pd-disaggregation-vs-chunked-prefill-kv-rdma-sizer-lab.html").exists()
