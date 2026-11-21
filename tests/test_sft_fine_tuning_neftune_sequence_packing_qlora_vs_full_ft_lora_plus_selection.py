import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestSFTFineTuningSequencePackingLigerLoRAPlusSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "sft_fine_tuning_neftune_sequence_packing_qlora_vs_full_ft_lora_plus_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "sft_fine_tuning_neftune_sequence_packing_qlora_vs_full_ft_lora_plus_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_LoRA_Plus_DoRA_Multipack_cu_seqlens_Plus_Liger_Unsloth"][
                "Effective_Useful_Tokens_Per_Sec_Per_GPU"
            ],
            tiers["Tier_1_Unpacked_Padded_Full_SFT_Unfused_CrossEntropy"][
                "Effective_Useful_Tokens_Per_Sec_Per_GPU"
            ],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "sft-fine-tuning-neftune-sequence-packing-qlora-vs-full-ft-lora-plus-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "sft-fine-tuning-neftune-sequence-packing-qlora-vs-full-ft-lora-plus-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "sft-fine-tuning-neftune-sequence-packing-qlora-vs-full-ft-lora-plus-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "sft-fine-tuning-neftune-sequence-packing-qlora-vs-full-ft-lora-plus-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "sft-fine-tuning-neftune-sequence-packing-qlora-vs-full-ft-lora-plus-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
