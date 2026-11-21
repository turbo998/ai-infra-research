import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestUltraEthernetUecSpectrumxPacketSprayingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "ultra_ethernet_uec_spectrumx_packet_spraying_ooo_congestion_control_selection"
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
        t1 = tiers["tier1_legacy_rocev2_static_ecmp_in_order_gbn_dcqcn"]
        t4 = tiers["tier4_spectrumx_or_uec_packet_spraying_plus_supernic_ooo_ddp_and_rtt_eqds_cc"]
        self.assertGreater(t4["multi_tenant_effective_busbw_gb_s"], 47.0)
        self.assertLess(
            t4["monthly_2048gpu_network_stall_tco_usd"],
            t1["monthly_2048gpu_network_stall_tco_usd"] * 0.20,
        )

    def test_artifacts_exist(self) -> None:
        slug = "ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
