from pathlib import Path
from experiments.enterprise_ide_code_completion_fim_debounce_cancel_gpu_sizing_selection.run import build_results


def test_enterprise_ide_code_completion_fim_debounce_cancel_gpu_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_ide_code_completion_fim_debounce_cancel_gpu_sizing_selection"
    tiers = res["ide_tiers"]
    assert len(tiers) == 4
    assert tiers[3]["fim_p99_e2e_latency_ms"] < 200.0
    assert tiers[3]["gpus_required_for_10k_developers"] < tiers[0]["gpus_required_for_10k_developers"] * 0.10
    assert tiers[3]["developer_suggestion_acceptance_rate_pct"] > 35.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-ide-code-completion-fim-debounce-cancel-gpu-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-ide-code-completion-fim-debounce-cancel-gpu-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/ide-code-completion-fim-debounce-gpu-sizer-lab.html").exists()
