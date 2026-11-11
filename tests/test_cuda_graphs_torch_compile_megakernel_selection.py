"""Unit tests for Round 45 GPU Kernel Execution Graph & Compilation Stack Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.cuda_graphs_torch_compile_megakernel_selection.run import (
    evaluate_cuda_graphs_torch_compile_megakernel_matrix,
)


class TestCudaGraphsTorchCompileMegakernelSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_cuda_graphs_torch_compile_megakernel_matrix()
        self.assertEqual(data["experiment_id"], "cuda_graphs_torch_compile_megakernel_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier1_pytorch_eager_cpu_launch_bound"]["cpu_launch_bubble_pct"],
            60.0,
        )
        self.assertLess(
            tiers["tier3_torch_compile_inductor_piecewise_cudagraph"]["graph_vram_overhead_gb"],
            tiers["tier2_monolithic_static_cuda_graphs_bucketed"]["graph_vram_overhead_gb"] * 0.2,
        )
        self.assertLess(
            tiers["tier4_persistent_megakernel_hopper_blackwell_pdl"]["decode_tpot_ms"],
            6.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_speedup_ratio"],
            4.5,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "cuda_graphs_torch_compile_megakernel_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_cuda_graphs_torch_compile_megakernel_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
