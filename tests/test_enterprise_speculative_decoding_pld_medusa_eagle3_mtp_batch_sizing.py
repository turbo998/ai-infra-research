from experiments.enterprise_speculative_decoding_pld_medusa_eagle3_mtp_batch_sizing.run import (
    evaluate_speculative_decoding_paradigms_sizing,
)


def test_speculative_decoding_paradigms_sizing():
    r = evaluate_speculative_decoding_paradigms_sizing()
    assert r["deliverable_id"] == "#637"
    assert r["theoretical_tau"] == 3.64
    assert r["measured_eagle3_mtp_tau"] == 3.45
    assert r["eagle3_mtp_tpot_ms"] == 9.0
    assert r["eagle3_mtp_speedup_x"] == 2.42
    assert r["auto_bypass_batch_threshold"] == 32
