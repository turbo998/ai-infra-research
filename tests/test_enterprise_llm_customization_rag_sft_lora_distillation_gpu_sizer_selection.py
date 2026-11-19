from __future__ import annotations

import unittest
from experiments.enterprise_llm_customization_rag_sft_lora_distillation_gpu_sizer_selection.run import build_payload


class TestEnterpriseLlmCustomizationRagSftLoraDistillationGpuSizerSelection(unittest.TestCase):
    def test_tier4_highest_accuracy_lowest_latency_and_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertGreater(tier4["domain_task_and_fresh_fact_accuracy_pct"], 96.0)
        self.assertGreater(tier4["structured_schema_and_style_adherence_pct"], 99.0)
        self.assertLess(tier4["p99_end_to_end_latency_ms"], tier2["p99_end_to_end_latency_ms"] * 0.2)
        self.assertLess(
            tier4["monthly_training_and_serving_tco_usd"],
            tier1["monthly_training_and_serving_tco_usd"] * 0.15,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
