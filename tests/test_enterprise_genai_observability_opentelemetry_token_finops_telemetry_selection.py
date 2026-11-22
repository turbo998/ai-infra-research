from pathlib import Path
from experiments.enterprise_genai_observability_opentelemetry_token_finops_telemetry_selection.run import build_results


def test_enterprise_genai_observability_opentelemetry_token_finops_telemetry_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_genai_observability_opentelemetry_token_finops_telemetry_selection"
    stacks = res["telemetry_stacks"]
    assert len(stacks) == 4
    assert stacks[3]["root_cause_mttr_minutes"] < 5.0
    assert stacks[3]["chargeback_attribution_accuracy_pct"] > 99.0
    assert stacks[3]["monthly_observability_bill_100m_reqs_usd"] < stacks[1]["monthly_observability_bill_100m_reqs_usd"] * 0.10

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-genai-observability-opentelemetry-token-finops-telemetry-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-genai-observability-opentelemetry-token-finops-telemetry-selection-faq.en.md").exists()
    assert (root / "docs/h5/genai-observability-otel-token-finops-lab.html").exists()
