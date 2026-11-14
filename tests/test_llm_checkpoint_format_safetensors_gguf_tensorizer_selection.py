"""Unit tests for Round 175 LLM Weight Serialization Format & Sub-Second Loading Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.llm_checkpoint_format_safetensors_gguf_tensorizer_selection.run import (
    evaluate_llm_checkpoint_format_safetensors_gguf_tensorizer_matrix,
)


class TestLlmCheckpointFormatSafetensorsGgufTensorizerSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_llm_checkpoint_format_safetensors_gguf_tensorizer_matrix()
        self.assertEqual(
            data["experiment_id"],
            "llm_checkpoint_format_safetensors_gguf_tensorizer_selection",
        )
        self.assertEqual(data["round"], 175)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertFalse(tiers["tier1_legacy_pytorch_pickle_bin"]["rce_security_safe"])
        self.assertTrue(tiers["tier3_safetensors_mmap_local_nvme"]["rce_security_safe"])
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_load_speedup_ratio"],
            50.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_host_ram_reduction_pct"],
            98.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "llm_checkpoint_format_safetensors_gguf_tensorizer_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_llm_checkpoint_format_safetensors_gguf_tensorizer_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
