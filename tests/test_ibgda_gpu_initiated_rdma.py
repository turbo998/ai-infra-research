import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.ibgda_gpu_initiated_rdma.run import RESULTS_PATH, build_results


class TestIbgdaGpuInitiatedRdma(unittest.TestCase):
    def test_build_results_matches_sealed_json(self) -> None:
        computed = build_results()
        sealed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(computed, sealed)

    def test_gdaki_doorbell_aggregation_and_cta_exclusive_qps(self) -> None:
        data = build_results()
        metrics = data["summary_metrics"]
        self.assertGreaterEqual(metrics["payload_512b_lat_speedup_x"], 7.5)
        self.assertGreaterEqual(metrics["payload_512b_bw_gain_x"], 16.0)
        self.assertEqual(metrics["qp25_cta_exclusive_dispatch_us"], 38.94)
        self.assertLess(
            metrics["qp25_cta_exclusive_dispatch_us"],
            metrics["qp5_gpu_sharing_dispatch_us"],
        )
        self.assertEqual(metrics["moe_60l_e2e_speedup_x"], 2.99)

    def test_cli_check_passes(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "ibgda_gpu_initiated_rdma" / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
