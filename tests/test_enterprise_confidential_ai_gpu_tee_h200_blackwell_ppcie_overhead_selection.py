from pathlib import Path
from experiments.enterprise_confidential_ai_gpu_tee_h200_blackwell_ppcie_overhead_selection.run import build_results


def test_enterprise_confidential_ai_gpu_tee_h200_blackwell_ppcie_overhead_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_confidential_ai_gpu_tee_h200_blackwell_ppcie_overhead_selection"
    tiers = res["tee_tiers"]
    assert len(tiers) == 4
    assert tiers[2]["e2e_inference_throughput_penalty_pct"] < 5.0
    assert tiers[3]["e2e_inference_throughput_penalty_pct"] < tiers[1]["e2e_inference_throughput_penalty_pct"]
    assert tiers[3]["cryptographic_remote_attestation"] is True

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-confidential-ai-gpu-tee-h200-blackwell-ppcie-overhead-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-confidential-ai-gpu-tee-h200-blackwell-ppcie-overhead-selection-faq.en.md").exists()
    assert (root / "docs/h5/confidential-ai-gpu-tee-ppcie-overhead-sizer-lab.html").exists()
