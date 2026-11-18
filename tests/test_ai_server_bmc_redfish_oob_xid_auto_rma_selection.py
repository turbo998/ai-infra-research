"""Unit tests for AI Server Out-of-Band Telemetry (BMC/Redfish), GPU XID Classification & Auto-RMA Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_server_bmc_redfish_oob_xid_auto_rma_selection.run import (
    evaluate_ai_server_bmc_redfish_oob_xid_auto_rma_matrix,
)


class TestAiServerBmcRedfishOobXidAutoRmaSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_server_bmc_redfish_oob_xid_auto_rma_matrix()
        self.assertEqual(data["experiment_id"], "ai_server_bmc_redfish_oob_xid_auto_rma_selection")
        self.assertEqual(data["round"], 179)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertEqual(
            tiers["tier4_oob_redfish_sse_tiered_auto_healing_rma"]["false_cordon_rate_on_xid31_pct"],
            0.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_mttr_speedup_ratio"],
            110.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_saved_gpu_hours_monthly"],
            17000.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_server_bmc_redfish_oob_xid_auto_rma_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_server_bmc_redfish_oob_xid_auto_rma_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
