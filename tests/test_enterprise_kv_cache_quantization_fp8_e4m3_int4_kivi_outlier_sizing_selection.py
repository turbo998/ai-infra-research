from pathlib import Path
from experiments.enterprise_kv_cache_quantization_fp8_e4m3_int4_kivi_outlier_sizing_selection.run import build_results


def test_enterprise_kv_cache_quantization_fp8_e4m3_int4_kivi_outlier_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_kv_cache_quantization_fp8_e4m3_int4_kivi_outlier_sizing_selection"
    fmts = res["kv_formats"]
    assert len(fmts) == 4
    assert fmts[2]["kv_bytes_per_token_kb"] == fmts[0]["kv_bytes_per_token_kb"] * 0.5
    assert fmts[3]["longbench_needle_accuracy_pct"] > 98.0
    assert fmts[3]["max_concurrent_32k_requests_2xh200"] > fmts[0]["max_concurrent_32k_requests_2xh200"] * 3

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-kv-cache-quantization-fp8-e4m3-int4-kivi-outlier-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-kv-cache-quantization-fp8-e4m3-int4-kivi-outlier-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/kv-cache-quantization-fp8-int4-kivi-sizer-lab.html").exists()
