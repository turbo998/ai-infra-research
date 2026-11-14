"""Unit tests for Round 177 NCCL Collective Algorithm, Protocol (LL/LL128/Simple) & Zero-SM CTA Tuning Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.nccl_algo_proto_ring_tree_nvls_cta_tuning_selection.run import (
    evaluate_nccl_algo_proto_ring_tree_nvls_cta_tuning_matrix,
)


class TestNcclAlgoProtoRingTreeNvlsCtaTuningSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_nccl_algo_proto_ring_tree_nvls_cta_tuning_matrix()
        self.assertEqual(data["experiment_id"], "nccl_algo_proto_ring_tree_nvls_cta_tuning_selection")
        self.assertEqual(data["round"], 177)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLessEqual(
            tiers["tier4_ub_zero_copy_nvls_ce_p2p_2sm_cga"]["nccl_sms_occupied"],
            4,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_sm_reduction_pct"],
            88.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_mfu_gain_pts"],
            15.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "nccl_algo_proto_ring_tree_nvls_cta_tuning_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_nccl_algo_proto_ring_tree_nvls_cta_tuning_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
