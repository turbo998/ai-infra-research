import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestSemanticCachingVsExactPrefixKVCacheSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "semantic_caching_redis_gptcache_exact_prefix_kv_cache_false_positive_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "semantic_caching_redis_gptcache_exact_prefix_kv_cache_false_positive_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Scoped_Entity_Verified_Semantic_Cache_Plus_Prefix_KV"][
                "False_Positive_Wrong_Answer_Rate_Pct"
            ],
            tiers["Tier_1_Unscoped_Gateway_Semantic_Cache_Cosine_090"][
                "False_Positive_Wrong_Answer_Rate_Pct"
            ],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "semantic-caching-redis-gptcache-exact-prefix-kv-cache-false-positive-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "semantic-caching-redis-gptcache-exact-prefix-kv-cache-false-positive-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "semantic-caching-redis-gptcache-exact-prefix-kv-cache-false-positive-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "semantic-caching-redis-gptcache-exact-prefix-kv-cache-false-positive-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "semantic-caching-redis-gptcache-exact-prefix-kv-cache-false-positive-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
