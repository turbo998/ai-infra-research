import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMDataCurationDedupNeMoCuratorRayDataSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_data_curation_dedup_minhash_sem_dedup_nemo_curator_ray_data_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_data_curation_dedup_minhash_sem_dedup_nemo_curator_ray_data_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Ray_Data_Heterogeneous_NeMo_Curator_cuDF_SemDeDup"]["Total_Curation_Wall_Clock_Hours"],
            tiers["Tier_1_Legacy_Apache_Spark_CPU_Cluster"]["Total_Curation_Wall_Clock_Hours"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-data-curation-dedup-minhash-sem-dedup-nemo-curator-ray-data-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-data-curation-dedup-minhash-sem-dedup-nemo-curator-ray-data-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-data-curation-dedup-minhash-sem-dedup-nemo-curator-ray-data-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-data-curation-dedup-minhash-sem-dedup-nemo-curator-ray-data-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-data-curation-dedup-minhash-sem-dedup-nemo-curator-ray-data-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
