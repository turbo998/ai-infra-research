"""Unit tests for Round 70 GPU HBM Fragmentation, CUDA VMM (expandable_segments) & vAttention Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.gpu_vmm_expandable_segments_vattention_defrag_selection.run import (
    evaluate_gpu_vmm_expandable_segments_vattention_matrix,
)


class TestGpuVmmExpandableSegmentsVattentionDefragSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_gpu_vmm_expandable_segments_vattention_matrix()
        self.assertEqual(
            data["experiment_id"],
            "gpu_vmm_expandable_segments_vattention_defrag_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_vmm_expandable_plus_vattention_mmu_kv_and_graph_sleep"]["usable_hbm_before_oom_gb"],
            78.0,
        )
        self.assertLess(
            tiers["tier3_cuda_vmm_expandable_segments_true"]["fragmented_stranded_hbm_pct"],
            5.0,
        )
        self.assertGreater(
            data["summary"]["tier4_reclaimed_hbm_per_gpu_gb"],
            12.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "gpu_vmm_expandable_segments_vattention_defrag_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_gpu_vmm_expandable_segments_vattention_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
