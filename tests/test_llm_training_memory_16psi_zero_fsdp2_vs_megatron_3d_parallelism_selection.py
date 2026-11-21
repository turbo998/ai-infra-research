import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmTrainingMemory16PsiZeroFsdp2VsMegatronSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_training_memory_16psi_zero_fsdp2_vs_megatron_3d_parallelism_selection"
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
        t1 = tiers["tier1_global_zero3_across_all_128_gpus_plus_full_activation_recompute"]
        t4 = tiers["tier4_megatron_tp_1f1b_pp_zero1_dp_plus_selective_activation_ckpt_fp8"]
        self.assertGreater(t4["achieved_training_mfu_pct"], 58.0)
        self.assertLess(t4["cross_node_communication_share_pct"], 8.0)
        self.assertLess(
            t4["cost_per_1t_training_tokens_usd"],
            t1["cost_per_1t_training_tokens_usd"] * 0.56,
        )

    def test_artifacts_exist(self) -> None:
        slug = "llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
