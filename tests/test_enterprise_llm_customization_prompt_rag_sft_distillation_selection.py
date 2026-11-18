"""Unit tests for Enterprise LLM Customization Path Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_llm_customization_prompt_rag_sft_distillation_selection.run import (
    evaluate_enterprise_llm_customization_prompt_rag_sft_distillation,
)


class TestEnterpriseLlmCustomizationPromptRagSftDistillationSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_llm_customization_prompt_rag_sft_distillation()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_llm_customization_prompt_rag_sft_distillation_selection",
        )
        self.assertEqual(data["round"], 363)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_hybrid_rag_plus_distilled_8b_slm_cascade_router"]["domain_accuracy_pct"],
            tiers["tier2_naive_knowledge_injection_lora_sft_without_rag"]["domain_accuracy_pct"],
        )
        self.assertLess(
            tiers["tier4_hybrid_rag_plus_distilled_8b_slm_cascade_router"]["monthly_tco_usd"],
            tiers["tier3_modular_hybrid_rag_plus_frontier_70b"]["monthly_tco_usd"],
        )
        self.assertGreater(
            data["summary"]["tier4_cost_savings_vs_tier3_frontier_rag_pct"],
            70.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_llm_customization_prompt_rag_sft_distillation_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_llm_customization_prompt_rag_sft_distillation()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
