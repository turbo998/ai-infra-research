"""Unit tests for Round 171 AI Datacenter GPU Hardware SKU, HBM Memory Wall & 3-Year TCO Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_gpu_sku_h100_h200_b200_gb200_mi325x_tco_selection.run import (
    evaluate_ai_gpu_sku_h100_h200_b200_gb200_mi325x_tco_matrix,
)


class TestAiGpuSkuH100H200B200Gb200Mi325xTcoSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_gpu_sku_h100_h200_b200_gb200_mi325x_tco_matrix()
        self.assertEqual(data["experiment_id"], "ai_gpu_sku_h100_h200_b200_gb200_mi325x_tco_selection")
        self.assertEqual(data["round"], 171)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertEqual(tiers["tier1_h100_80gb_sxm5_two_node_tp16"]["min_nodes_for_671b_fp8"], 2)
        self.assertEqual(tiers["tier2_h200_141gb_mi325x_256gb_single_node"]["min_nodes_for_671b_fp8"], 1)
        self.assertGreater(
            tiers["tier4_gb200_nvl72_liquid_rack_supernode"]["throughput_tok_per_gpu_s"],
            1200.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_token_cost_reduction_pct"],
            85.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_gpu_sku_h100_h200_b200_gb200_mi325x_tco_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_gpu_sku_h100_h200_b200_gb200_mi325x_tco_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
