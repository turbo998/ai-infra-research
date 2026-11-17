"""Unit tests for Round 375 Enterprise Coding Copilot, FIM Dual-Track Serving, AST Repo-Map & Sandboxed SWE-Agent Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_coding_copilot_fim_ast_repomap_sandbox_selection.run import (
    evaluate_enterprise_coding_copilot_fim_ast_repomap_sandbox,
)


class TestEnterpriseCodingCopilotFimAstRepomapSandboxSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_coding_copilot_fim_ast_repomap_sandbox()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_coding_copilot_fim_ast_repomap_sandbox_selection",
        )
        self.assertEqual(data["round"], 375)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLessEqual(
            tiers["tier4_dual_track_ast_repomap_plus_sandbox_compiler_test_loop"]["inline_fim_p99_latency_ms"],
            220.0,
        )
        self.assertGreater(
            tiers["tier4_dual_track_ast_repomap_plus_sandbox_compiler_test_loop"]["multifile_compile_pass_rate_pct"],
            90.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_coding_copilot_fim_ast_repomap_sandbox_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_coding_copilot_fim_ast_repomap_sandbox()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
