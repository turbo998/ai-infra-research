from pathlib import Path
from experiments.enterprise_multimodal_vlm_epd_disaggregation_visual_token_cache_sizing_selection.run import build_results


def test_enterprise_multimodal_vlm_epd_disaggregation_visual_token_cache_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_multimodal_vlm_epd_disaggregation_visual_token_cache_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["p95_ttft_ms"] < tiers[2]["p95_ttft_ms"] < tiers[0]["p95_ttft_ms"]
    assert tiers[3]["cluster_goodput_req_s"] > tiers[0]["cluster_goodput_req_s"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-multimodal-vlm-epd-disaggregation-visual-token-cache-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-multimodal-vlm-epd-disaggregation-visual-token-cache-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/multimodal-vlm-epd-disaggregation-visual-token-sizer-lab.html").exists()
