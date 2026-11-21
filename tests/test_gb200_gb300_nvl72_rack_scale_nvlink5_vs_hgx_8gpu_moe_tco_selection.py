import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestGb200Gb300Nvl72VsHgx8GpuSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "gb200_gb300_nvl72_rack_scale_nvlink5_vs_hgx_8gpu_moe_tco_selection"
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
        t2 = tiers["tier2_nine_8gpu_hgx_h200_nodes_with_deepep_ibgda"]
        t4 = tiers["tier4_single_gb200_gb300_nvl72_rack_72way_nvlink5_plus_grace_c2c"]
        self.assertGreater(t4["per_gpu_goodput_tps"], t2["per_gpu_goodput_tps"] * 3.0)
        self.assertLess(t4["achieved_p99_tpot_ms"], 6.0)

    def test_artifacts_exist(self) -> None:
        slug = "gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
