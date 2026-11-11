"""Unit tests for Round 56 LLM Output Watermarking, SynthID & Entropy-Gated Provenance Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.llm_watermarking_cryptographic_provenance_selection.run import (
    evaluate_llm_watermarking_cryptographic_provenance_matrix,
)


class TestLlmWatermarkingCryptographicProvenanceSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_llm_watermarking_cryptographic_provenance_matrix()
        self.assertEqual(data["experiment_id"], "llm_watermarking_cryptographic_provenance_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_gpu_fused_entropy_gated_synthid_speculative_c2pa"]["code_pass_at_1_pct"],
            tiers["tier2_naive_kgw_green_red_list_logit_bias"]["code_pass_at_1_pct"] + 15.0,
        )
        self.assertGreater(
            tiers["tier4_gpu_fused_entropy_gated_synthid_speculative_c2pa"]["tpr_at_0_1pct_fpr_clean"],
            98.0,
        )
        self.assertLess(
            tiers["tier4_gpu_fused_entropy_gated_synthid_speculative_c2pa"]["sampling_overhead_ms_per_token"],
            0.15,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "llm_watermarking_cryptographic_provenance_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_llm_watermarking_cryptographic_provenance_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
