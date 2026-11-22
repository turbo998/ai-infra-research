from pathlib import Path
from experiments.enterprise_structured_output_xgrammar_llguidance_json_schema_latency_selection.run import build_results


def test_enterprise_structured_output_xgrammar_llguidance_json_schema_latency_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_structured_output_xgrammar_llguidance_json_schema_latency_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["per_token_mask_overhead_ms"] < tiers[1]["per_token_mask_overhead_ms"]
    assert tiers[3]["effective_goodput_tok_s"] > tiers[1]["effective_goodput_tok_s"] > tiers[0]["effective_goodput_tok_s"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-structured-output-xgrammar-llguidance-json-schema-latency-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-structured-output-xgrammar-llguidance-json-schema-latency-selection-faq.en.md").exists()
    assert (root / "docs/h5/structured-output-xgrammar-llguidance-latency-lab.html").exists()
