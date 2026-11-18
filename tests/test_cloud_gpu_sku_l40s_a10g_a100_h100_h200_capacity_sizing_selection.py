"""Unit tests for Cloud GPU Instance SKU Selection & Concurrency Capacity Sizer Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.cloud_gpu_sku_l40s_a10g_a100_h100_h200_capacity_sizing_selection.run import (
    evaluate_cloud_gpu_sku_l40s_a10g_a100_h100_h200_capacity_sizing,
)


class TestCloudGpuSkuL40sA10gA100H100H200CapacitySizingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_cloud_gpu_sku_l40s_a10g_a100_h100_h200_capacity_sizing()
        self.assertEqual(
            data["experiment_id"],
            "cloud_gpu_sku_l40s_a10g_a100_h100_h200_capacity_sizing_selection",
        )
        self.assertEqual(data["round"], 364)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_flagship_h200_141gb_or_h100_fp8_70b_king"]["max_concurrent_8k_sessions"],
            tiers["tier2_legacy_2x_a100_80gb_bf16_trap"]["max_concurrent_8k_sessions"],
        )
        self.assertLess(
            tiers["tier4_flagship_h200_141gb_or_h100_fp8_70b_king"]["cost_per_1m_tokens_usd"],
            tiers["tier2_legacy_2x_a100_80gb_bf16_trap"]["cost_per_1m_tokens_usd"],
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_token_cost_reduction_pct"],
            70.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "cloud_gpu_sku_l40s_a10g_a100_h100_h200_capacity_sizing_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_cloud_gpu_sku_l40s_a10g_a100_h100_h200_capacity_sizing()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
