"""Unit tests for Round 59 LLM Pretraining Optimizer Architecture & Matrix Orthogonalization Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.distributed_optimizer_adamw_8bit_muon_shampoo_selection.run import (
    evaluate_distributed_optimizer_muon_adamw_matrix,
)


class TestDistributedOptimizerAdamw8BitMuonShampooSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_distributed_optimizer_muon_shampoo()
        self.assertEqual(
            data["experiment_id"],
            "distributed_optimizer_adamw_8bit_muon_shampoo_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_distributed_muon_newton_schulz5_hybrid_adamw"]["token_sample_efficiency_x"],
            1.30,
        )
        self.assertEqual(
            tiers["tier4_distributed_muon_newton_schulz5_hybrid_adamw"]["optimizer_state_bytes_per_param"],
            6.0,
        )
        self.assertGreater(
            data["summary"]["tier4_net_savings_vs_tier1_usd"],
            1000000.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "distributed_optimizer_adamw_8bit_muon_shampoo_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_distributed_optimizer_muon_adamw_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


def evaluate_distributed_optimizer_muon_shampoo() -> dict[str, Any]:
    return evaluate_distributed_optimizer_muon_adamw_matrix()


if __name__ == "__main__":
    unittest.main()
