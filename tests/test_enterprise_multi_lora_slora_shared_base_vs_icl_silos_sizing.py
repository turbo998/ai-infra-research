from experiments.enterprise_multi_lora_slora_shared_base_vs_icl_silos_sizing.run import (
    evaluate_multi_lora_slora_shared_base_sizing,
)


def test_multi_lora_slora_shared_base_sizing():
    r = evaluate_multi_lora_slora_shared_base_sizing()
    assert r["deliverable_id"] == "#636"
    assert r["slora_shared_h200_gpus"] == 8
    assert r["gpu_savings_vs_silos_pct"] == 96.9
    assert r["annual_savings_vs_silos_usd"] == 9523200
    assert r["slora_domain_accuracy_pct"] == 97.6
    assert r["slora_p99_tpot_ms"] == 14.8
