from experiments.enterprise_post_training_16psi_lora_grpo_verl_shared_cluster.run import (
    evaluate_post_training_16psi_lora_grpo_sizing,
)


def test_post_training_16psi_lora_grpo_sizing():
    r = evaluate_post_training_16psi_lora_grpo_sizing()
    assert r["deliverable_id"] == "#639"
    assert r["full_sft_total_vram_gb"] == 1300
    assert r["lora_sft_total_vram_gb"] == 156
    assert r["vram_savings_pct"] == 88.0
    assert r["verl_resharding_sec"] == 0.9
    assert r["grpo_iteration_speedup_x"] == 2.45
    assert r["annual_shared_cluster_savings_usd"] == 441600
