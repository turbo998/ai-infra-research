import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmInferencePagedAttentionFlashInferCudaGraphSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_inference_pagedattention_flashinfer_cudagraph_memory_pool_selection"
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
        t1 = tiers["tier1_contiguous_kv_allocation_eager_mode_without_cudagraph"]
        t4 = tiers["tier4_flashinfer_cascade_paged_kv_piecewise_cudagraph_persistent_cache"]
        self.assertGreater(t4["effective_kv_hbm_utilization_pct"], 95.0)
        self.assertLess(t4["cpu_kernel_launch_overhead_per_step_ms"], 0.5)
        self.assertLess(
            t4["cost_per_1m_tokens_usd"],
            t1["cost_per_1m_tokens_usd"] * 0.30,
        )

    def test_artifacts_exist(self) -> None:
        slug = "llm-inference-pagedattention-flashinfer-cudagraph-memory-pool"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
