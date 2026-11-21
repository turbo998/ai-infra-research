import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestIDECodeCompletionFIMCodingAgentSpeculativeEditsSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "ide_code_completion_fim_coding_agent_speculative_edits_cursor_rag_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "ide_code_completion_fim_coding_agent_speculative_edits_cursor_rag_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Next_Edit_FIM_Abort_Plus_Speculative_Fast_Apply_AST_Merkle"]["Full_File_4800Tok_Apply_Latency_sec"],
            tiers["Tier_1_Unified_70B_Pool_For_Both_Tab_And_Chat"]["Full_File_4800Tok_Apply_Latency_sec"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "ide-code-completion-fim-coding-agent-speculative-edits-cursor-rag-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "ide-code-completion-fim-coding-agent-speculative-edits-cursor-rag-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "ide-code-completion-fim-coding-agent-speculative-edits-cursor-rag-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "ide-code-completion-fim-coding-agent-speculative-edits-cursor-rag-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "ide-code-completion-fim-coding-agent-speculative-edits-cursor-rag-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
