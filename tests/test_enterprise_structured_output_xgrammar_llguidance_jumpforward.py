from experiments.enterprise_structured_output_xgrammar_llguidance_jumpforward.run import (
    compute_jump_forward_tokens_per_step,
    evaluate_structured_output_stacks,
)


def test_enterprise_structured_output_xgrammar_llguidance_jumpforward() -> None:
    res = evaluate_structured_output_stacks()
    assert res["Deliverable_ID"] == "#642"
    assert compute_jump_forward_tokens_per_step(0.54) == 2.17
    jf = res["Stacks"]["SGLang_XGrammar_llguidance_JumpForward"]
    assert jf["mask_latency_ms"] == 0.04
    assert jf["effective_json_tps"] == 7680
    assert jf["speedup_vs_standard"] == 1.85
    assert jf["required_h200_gpus_200rps"] == 8
