import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestGPUPowerCappingDVFSRackPDUOversubscriptionSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "gpu_power_capping_dvfs_transient_spike_rack_pdu_oversubscription_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "gpu_power_capping_dvfs_transient_spike_rack_pdu_oversubscription_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_Workload_Aware_DVFS_520W_Decode_580W_Train_Plus_BMC_Telemetry_Loop"]["Deployable_H200_GPUs_In_10MW"],
            tiers["Tier_1_Uncapped_700W_Nameplate_Static_Provisioning"]["Deployable_H200_GPUs_In_10MW"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "gpu-power-capping-dvfs-transient-spike-rack-pdu-oversubscription-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "gpu-power-capping-dvfs-transient-spike-rack-pdu-oversubscription-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "gpu-power-capping-dvfs-transient-spike-rack-pdu-oversubscription-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "gpu-power-capping-dvfs-transient-spike-rack-pdu-oversubscription-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "gpu-power-capping-dvfs-transient-spike-rack-pdu-oversubscription-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
