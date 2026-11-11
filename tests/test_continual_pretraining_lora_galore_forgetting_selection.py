"""Unit tests for Round 50 Continual Pre-Training (CPT) & Catastrophic Forgetting Prevention Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.continual_pretraining_lora_galore_forgetting_selection.run import (
    evaluate_continual_pretraining_lora_galore_forgetting_matrix,
)


class TestContinualPretrainingLoraGaloreForgettingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_continual_pretraining_lora_galore_forgetting_matrix()
        self.assertEqual(data["experiment_id"], "continual_pretraining_lora_galore_forgetting_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_galore_lowrank_grad_replay_dare_ties_merge"]["domain_knowledge_accuracy_pct"],
            tiers["tier2_peft_qlora_dora_rank64_adapter"]["domain_knowledge_accuracy_pct"] + 20.0,
        )
        self.assertGreater(
            tiers["tier4_galore_lowrank_grad_replay_dare_ties_merge"]["general_reasoning_retention_pct"],
            85.0,
        )
        self.assertLess(
            tiers["tier4_galore_lowrank_grad_replay_dare_ties_merge"]["total_cluster_vram_gb"],
            tiers["tier3_full_cpt_25pct_replay_wsd_decoupled_lr"]["total_cluster_vram_gb"] * 0.3,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "continual_pretraining_lora_galore_forgetting_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_continual_pretraining_lora_galore_forgetting_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
