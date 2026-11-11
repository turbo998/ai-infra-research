"""Unit tests for Round 67 Discrete Diffusion LLM (Mercury / LLaDA / Block Diffusion BD3-LM) Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.diffusion_llm_mercury_llada_parallel_decoding_selection.run import (
    evaluate_diffusion_llm_mercury_llada_matrix,
)


class TestDiffusionLlmMercuryLladaParallelDecodingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_diffusion_llm_mercury_llada_matrix()
        self.assertEqual(
            data["experiment_id"],
            "diffusion_llm_mercury_llada_parallel_decoding_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_block_diffusion_kv_cache_confidence_early_exit_mercury"]["generation_throughput_tok_per_s"],
            1000.0,
        )
        self.assertLess(
            tiers["tier4_block_diffusion_kv_cache_confidence_early_exit_mercury"]["hbm_weight_reads_per_512_tokens"],
            35,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_throughput_speedup_ratio"],
            10.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "diffusion_llm_mercury_llada_parallel_decoding_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_diffusion_llm_mercury_llada_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
