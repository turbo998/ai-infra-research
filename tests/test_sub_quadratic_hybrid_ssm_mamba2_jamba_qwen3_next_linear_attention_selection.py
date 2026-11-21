import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestSubQuadraticHybridSSMMamba2JambaSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "sub_quadratic_hybrid_ssm_mamba2_jamba_qwen3_next_linear_attention_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "sub_quadratic_hybrid_ssm_mamba2_jamba_qwen3_next_linear_attention_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Hybrid_SSM_Gated_DeltaNet_7To1_GQA_Mamba2"]["State_Plus_KV_Memory_Per_256K_Req_GB"],
            tiers["Tier_1_Pure_Transformer_All_64_GQA_Layers"]["State_Plus_KV_Memory_Per_256K_Req_GB"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "sub-quadratic-hybrid-ssm-mamba2-jamba-qwen3-next-linear-attention-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "sub-quadratic-hybrid-ssm-mamba2-jamba-qwen3-next-linear-attention-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "sub-quadratic-hybrid-ssm-mamba2-jamba-qwen3-next-linear-attention-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "sub-quadratic-hybrid-ssm-mamba2-jamba-qwen3-next-linear-attention-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "sub-quadratic-hybrid-ssm-mamba2-jamba-qwen3-next-linear-attention-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
