import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseLlmContextCompressionSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_llm_context_compression_history_pruning_prefix_cache_budget_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_script_check_and_invariants(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"], "synthetic + anonymized_reported")
        self.assertFalse(data["independently_reproduced"])
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unbounded_full_history_append"]
        t2 = tiers["tier2_per_turn_sliding_window_fifo_drop"]
        t4 = tiers["tier4_epoch_block_compaction_tool_slimming_pinned_ledger"]
        self.assertGreater(t4["kv_prefix_cache_hit_rate_pct"], 88.0)
        self.assertLess(t2["kv_prefix_cache_hit_rate_pct"], 20.0)
        self.assertGreater(t4["turn20_constraint_retention_pct"], 98.0)
        self.assertLess(
            t4["monthly_token_and_kv_gpu_tco_usd"],
            t1["monthly_token_and_kv_gpu_tco_usd"] * 0.12,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-llm-context-compression-history-pruning-prefix-cache-budget"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
