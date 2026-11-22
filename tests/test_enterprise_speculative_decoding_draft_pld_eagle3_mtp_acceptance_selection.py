from pathlib import Path
from experiments.enterprise_speculative_decoding_draft_pld_eagle3_mtp_acceptance_selection.run import build_results


def test_enterprise_speculative_decoding_draft_pld_eagle3_mtp_acceptance_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_speculative_decoding_draft_pld_eagle3_mtp_acceptance_selection"
    methods = res["spec_methods"]
    assert len(methods) == 4
    assert methods[3]["mean_accepted_tokens_per_step_tau"] > methods[0]["mean_accepted_tokens_per_step_tau"]
    assert methods[3]["batch_8_tpot_speedup_x"] > 2.3

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-speculative-decoding-draft-pld-eagle3-mtp-acceptance-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-speculative-decoding-draft-pld-eagle3-mtp-acceptance-selection-faq.en.md").exists()
    assert (root / "docs/h5/speculative-decoding-draft-pld-eagle3-mtp-lab.html").exists()
