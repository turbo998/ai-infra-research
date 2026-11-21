import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestPrefillDecodeDisaggregationBoundarySelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection"
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
        t2 = tiers["tier2_colocated_chunked_prefill_sarathi_vllm"]
        t4 = tiers["tier4_pd_disaggregation_rdma_nixl_mooncake_layerwise_kv_push"]
        self.assertLess(t4["p99_tpot_ms"], 18.0)
        self.assertGreater(t4["sla_compliant_goodput_tok_per_s_gpu"], t2["sla_compliant_goodput_tok_per_s_gpu"] * 2.0)
        self.assertLess(t4["cost_per_1m_sla_tokens_usd"], t2["cost_per_1m_sla_tokens_usd"] * 0.60)

    def test_artifacts_exist(self) -> None:
        slug = "prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
