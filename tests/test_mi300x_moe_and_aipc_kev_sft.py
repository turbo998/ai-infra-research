import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestMi300xLargeMoeAndAipcKevSft(unittest.TestCase):
    def test_experiment_suite_check(self) -> None:
        script = ROOT / "experiments" / "mi300x_large_moe_aipc_kev_sft" / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"Suite failed:\n{proc.stderr}\n{proc.stdout}")

    def test_bilingual_docs_and_7card_reports_exist(self) -> None:
        zh_doc = ROOT / "docs" / "components" / "mi300x-large-moe-bf16-mxfp4-and-aipc-decision-sft.md"
        en_doc = ROOT / "docs" / "components" / "mi300x-large-moe-bf16-mxfp4-and-aipc-decision-sft.en.md"
        mi300x_cards = ROOT / "docs" / "reports" / "mi300x-large-moe-bf16-mxfp4-cards.html"
        aipc_cards = ROOT / "docs" / "reports" / "aipc-kev-sft-q8-gguf-cards.html"

        for path in (zh_doc, en_doc, mi300x_cards, aipc_cards):
            self.assertTrue(path.is_file(), f"Missing expected artifact: {path}")

        for html_path in (mi300x_cards, aipc_cards):
            html = html_path.read_text(encoding="utf-8")
            self.assertEqual(html.count('class="report-card"'), 7)
            self.assertIn("Content-Security-Policy", html)
            self.assertNotIn("<script", html)


if __name__ == "__main__":
    unittest.main()
