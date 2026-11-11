"""Unit tests for Round 49 Heterogeneous GPU (NVIDIA + AMD MI300X) & Triton/AITER Stack Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.heterogeneous_gpu_rocm_cuda_triton_stack_selection.run import (
    evaluate_heterogeneous_gpu_rocm_cuda_triton_matrix,
)


class TestHeterogeneousGpuRocmCudaTritonStackSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_heterogeneous_gpu_rocm_cuda_triton_matrix()
        self.assertEqual(data["experiment_id"], "heterogeneous_gpu_rocm_cuda_triton_stack_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier3_triton_tilelang_aiter_xcd_swizzled_monolithic"]["node_throughput_tok_s"],
            tiers["tier1_naive_hipify_unoptimized_rocm_port"]["node_throughput_tok_s"] * 2.4,
        )
        self.assertGreater(
            tiers["tier4_heterogeneous_pd_disaggregation_h100_prefill_mi300x_decode"]["node_throughput_tok_s"],
            3500.0,
        )
        self.assertGreater(
            data["summary"]["tier4_cost_reduction_vs_tier1_pct"],
            70.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "heterogeneous_gpu_rocm_cuda_triton_stack_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_heterogeneous_gpu_rocm_cuda_triton_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
