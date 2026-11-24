from experiments.enterprise_post_training_sft_lora_grpo_shared_cluster_sizing.run import evaluate_post_training_sizing


def test_evaluate_post_training_sizing():
    r = evaluate_post_training_sizing(70)
    assert r["full_sft_vram_gb"] == 1300
    assert r["lora_sft_vram_gb"] == 156
    assert r["qlora_sft_vram_gb"] == 52
    assert r["full_sft_h100_gpus"] == 16
    assert r["lora_sft_h200_gpus"] == 2
    assert r["vram_reduction_lora_vs_full_pct"] == 88.0
