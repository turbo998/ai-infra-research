import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmAttentionArchitectureMhaGqaMlaSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_attention_architecture_mha_gqa_mla_matrix_absorption_hybrid_linear_selection"
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
        t2 = tiers["tier2_standard_gqa_8_kv_heads_llama3_qwen25"]
        t4 = tiers["tier4_mla_with_weight_matrix_absorption_flashmla_and_7to1_hybrid_linear"]
        self.assertLess(t4["kv_cache_kb_per_token_fp8"], t2["kv_cache_kb_per_token_fp8"] * 0.25)
        self.assertGreater(t4["max_concurrent_32k_sessions_per_node"], 200)

    def test_artifacts_exist(self) -> None:
        slug = "llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
