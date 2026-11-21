import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMSecurityIndirectPromptInjectionSpotlightingCaMeLSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_security_indirect_prompt_injection_spotlighting_camel_privilege_isolation_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_security_indirect_prompt_injection_spotlighting_camel_privilege_isolation_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_CaMeL_Dual_LLM_Control_Data_Separation_Plus_Taint_Policy"][
                "Indirect_Injection_Attack_Success_Rate_ASR_Pct"
            ],
            tiers["Tier_1_Flat_Prompt_Concatenation_Unrestricted_Tools"][
                "Indirect_Injection_Attack_Success_Rate_ASR_Pct"
            ],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
