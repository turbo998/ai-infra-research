from pathlib import Path
from experiments.enterprise_llm_overload_admission_control_kv_watermark_graceful_degradation_selection.run import build_results


def test_enterprise_llm_overload_admission_control_kv_watermark_graceful_degradation_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_llm_overload_admission_control_kv_watermark_graceful_degradation_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["goodput_sla_attainment_pct"] > tiers[2]["goodput_sla_attainment_pct"] > tiers[0]["goodput_sla_attainment_pct"]
    assert tiers[3]["p99_ttft_ms"] < tiers[0]["p99_ttft_ms"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-llm-overload-admission-control-kv-watermark-graceful-degradation-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-llm-overload-admission-control-kv-watermark-graceful-degradation-selection-faq.en.md").exists()
    assert (root / "docs/h5/llm-overload-admission-control-kv-watermark-degradation-lab.html").exists()
