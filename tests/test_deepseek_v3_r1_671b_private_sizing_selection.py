"""Unit tests for Round 172 Enterprise DeepSeek-V3 / R1 (671B Full MoE) Private Deployment Sizing Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.deepseek_v3_r1_671b_private_sizing_selection.run import (
    evaluate_deepseek_v3_r1_671b_private_sizing_matrix,
)


class TestDeepseekV3R1671bPrivateSizingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_deepseek_v3_r1_671b_private_sizing_matrix()
        self.assertEqual(data["experiment_id"], "deepseek_v3_r1_671b_private_sizing_selection")
        self.assertEqual(data["round"], 172)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier3_single_node_8xh200_mi300x_fp8_mtp"]["total_throughput_tok_s"],
            tiers["tier2_two_node_16xh100_80gb_colocated_tp16"]["total_throughput_tok_s"],
        )
        self.assertGreater(
            data["summary"]["tier3_capex_savings_vs_tier2_usd"],
            200000.0,
        )
        self.assertGreater(
            data["summary"]["tier4_token_cost_reduction_vs_tier2_pct"],
            75.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "deepseek_v3_r1_671b_private_sizing_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_deepseek_v3_r1_671b_private_sizing_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
