import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestMultimodalVlmVideoLlmEpdDisaggregationSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "multimodal_vlm_video_llm_anyres_token_budget_epd_disaggregation_selection"
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
        t1 = tiers["tier1_monolithic_colocated_vit_prefill_decode_unbounded_4k_tiles"]
        t4 = tiers["tier4_3stage_epd_disaggregation_plus_sha256_visual_embedding_and_kv_cache"]
        self.assertLess(t4["p95_ttft_ms"], 500.0)
        self.assertLess(t4["p99_decode_tpot_ms"], 16.0)
        self.assertLess(
            t4["cost_per_1000_multimodal_queries_usd"],
            t1["cost_per_1000_multimodal_queries_usd"] * 0.25,
        )

    def test_artifacts_exist(self) -> None:
        slug = "multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
