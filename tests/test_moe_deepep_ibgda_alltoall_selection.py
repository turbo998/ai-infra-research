"""Unit tests for Round 46 MoE DeepEP & GPU-Initiated RDMA (IBGDA) All-to-All Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.moe_deepep_ibgda_alltoall_selection.run import (
    evaluate_moe_deepep_ibgda_alltoall_matrix,
)


class TestMoeDeepepIbgdaAlltoallSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_moe_deepep_ibgda_alltoall_matrix()
        self.assertEqual(data["experiment_id"], "moe_deepep_ibgda_alltoall_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier1_standard_nccl_host_proxy_alltoall"]["decode_dispatch_combine_us_per_layer"],
            tiers["tier4_deepep_low_latency_ibgda_hook_overlap_eplb"]["decode_dispatch_combine_us_per_layer"] * 5.5,
        )
        self.assertEqual(
            tiers["tier4_deepep_low_latency_ibgda_hook_overlap_eplb"]["gpu_sms_used_for_comm"],
            0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_tpot_speedup"],
            4.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "moe_deepep_ibgda_alltoall_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_moe_deepep_ibgda_alltoall_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
