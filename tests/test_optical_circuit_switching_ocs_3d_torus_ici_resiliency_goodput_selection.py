import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestOpticalCircuitSwitchingOcsIciResiliencySelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "optical_circuit_switching_ocs_3d_torus_ici_resiliency_goodput_selection"
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
        t1 = tiers["tier1_statically_cabled_3d_torus_no_ocs_no_ici_resiliency"]
        t2 = tiers["tier2_3tier_electrical_packet_clos_fat_tree_oeo_optics"]
        t4 = tiers["tier4_palomar_mems_ocs_ici_resiliency_twisted_torus_spare_cubes"]
        self.assertGreater(t4["superpod_effective_goodput_pct"], 99.0)
        self.assertLess(t4["topology_fragmentation_stranded_chips_pct"], 1.0)
        self.assertLess(t4["fabric_switch_and_optics_power_kw"], t2["fabric_switch_and_optics_power_kw"] * 0.15)
        self.assertLess(
            t4["monthly_9216_chip_interconnect_and_downtime_tco_usd"],
            t1["monthly_9216_chip_interconnect_and_downtime_tco_usd"] * 0.30,
        )

    def test_artifacts_exist(self) -> None:
        slug = "optical-circuit-switching-ocs-3d-torus-ici-resiliency-goodput"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
