import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestDistributedKVCachePoolLMCacheMooncakeSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "distributed_kv_cache_pool_lmcache_mooncake_vllm_production_stack_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "distributed_kv_cache_pool_lmcache_mooncake_vllm_production_stack_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Mooncake_Dynamo_KV_Router_RDMA_Layer_Pipelined"]["Multi_Turn_32K_P95_TTFT_ms"],
            tiers["Tier_1_Isolated_Pod_HBM_Cache_Round_Robin"]["Multi_Turn_32K_P95_TTFT_ms"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "distributed-kv-cache-pool-lmcache-mooncake-vllm-production-stack-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "distributed-kv-cache-pool-lmcache-mooncake-vllm-production-stack-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "distributed-kv-cache-pool-lmcache-mooncake-vllm-production-stack-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "distributed-kv-cache-pool-lmcache-mooncake-vllm-production-stack-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "distributed-kv-cache-pool-lmcache-mooncake-vllm-production-stack-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
