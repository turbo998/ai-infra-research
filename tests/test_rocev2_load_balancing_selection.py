import json
import pathlib
import subprocess
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "rocev2_load_balancing_selection"
RUN_PY = EXP_DIR / "run.py"
RESULTS_JSON = EXP_DIR / "results.json"


class TestRoCEv2LoadBalancingSelection(unittest.TestCase):
    def test_run_and_check_deterministic_results(self) -> None:
        subprocess.run([sys.executable, str(RUN_PY)], check=True)
        subprocess.run([sys.executable, str(RUN_PY), "--check"], check=True)
        data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

        self.assertEqual(data["provenance"]["data_classification"], "synthetic")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(data["provenance"]["live_gpu_switch_fabric_execution"], "not_measured")

        s = data["strategies"]
        # Option A: Static ECMP
        self.assertEqual(s["ecmp_pfc_dcqcn"]["allreduce_busbw_gb_s"], 27.0)
        self.assertEqual(s["ecmp_pfc_dcqcn"]["allreduce_busbw_gbps"], 216.0)
        self.assertEqual(s["ecmp_pfc_dcqcn"]["moe_alltoall_p99_ms"], 42.0)
        self.assertEqual(s["ecmp_pfc_dcqcn"]["cluster_goodput_pct"], 78.4)

        # Option B: Multi-QP Scaling
        self.assertEqual(s["ecmp_qp_scaling"]["allreduce_busbw_gb_s"], 37.0)
        self.assertEqual(s["ecmp_qp_scaling"]["busbw_speedup_vs_static_ecmp"], 1.37)
        self.assertEqual(s["ecmp_qp_scaling"]["moe_alltoall_p99_ms"], 26.5)

        # Option C: Flowlet DLB
        self.assertEqual(s["flowlet_dlb"]["allreduce_busbw_gb_s"], 42.5)
        self.assertEqual(s["flowlet_dlb"]["busbw_speedup_vs_static_ecmp"], 1.57)
        self.assertEqual(s["flowlet_dlb"]["moe_alltoall_p99_ms"], 18.4)

        # Option D: Packet Spraying + OoO RDMA
        self.assertEqual(s["packet_spraying_ooo_rdma"]["allreduce_busbw_gb_s"], 47.5)
        self.assertEqual(s["packet_spraying_ooo_rdma"]["allreduce_busbw_gbps"], 380.0)
        self.assertEqual(s["packet_spraying_ooo_rdma"]["busbw_speedup_vs_static_ecmp"], 1.76)
        self.assertEqual(s["packet_spraying_ooo_rdma"]["moe_alltoall_p99_ms"], 11.2)
        self.assertEqual(s["packet_spraying_ooo_rdma"]["moe_alltoall_speedup_vs_static_ecmp"], 3.75)
        self.assertEqual(s["packet_spraying_ooo_rdma"]["cluster_goodput_pct"], 98.2)
        self.assertEqual(s["packet_spraying_ooo_rdma"]["monthly_savings_vs_static_ecmp_usd"], 145616.49)


if __name__ == "__main__":
    unittest.main()
