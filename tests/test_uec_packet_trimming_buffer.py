import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.uec_packet_trimming_buffer.run import (
    SONIC_INGRESS_POOL_BYTES,
    build_results,
    compute_dynamic_threshold_absorption,
)


class TestUecPacketTrimmingBuffer(unittest.TestCase):
    def test_sonic_dynamic_threshold_formula(self) -> None:
        # When alpha = 1.0 (alpha_exp = 0) and 4 active queues with 0 reserved headroom,
        # each queue receives 1 / (1 + 4*1) = 20% of pool, total absorbed = 80.0%
        alloc = compute_dynamic_threshold_absorption(
            SONIC_INGRESS_POOL_BYTES, alpha_exp=0, active_congested_queues=4, reserved_headroom_bytes=0
        )
        self.assertEqual(alloc["pool_utilization_pct"], 80.0)

    def test_incast_pfc_vs_uec_trimming(self) -> None:
        res = build_results()
        self.assertAlmostEqual(
            res["sonic_config_baseline"]["roce_800g_32port_xoff_headroom_reserve_pct"], 38.13, places=2
        )
        incast_32 = [r for r in res["incast_comparison"] if r["incast_senders"] == 32][0]
        incast_64 = [r for r in res["incast_comparison"] if r["incast_senders"] == 64][0]
        self.assertGreaterEqual(incast_32["uec_cct_speedup_x"], 4.5)
        self.assertGreaterEqual(incast_64["victim_tail_reduction_x"], 40.0)

    def test_cli_check_mode(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "uec_packet_trimming_buffer" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        committed = json.loads((script.parent / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(committed["experiment"], "uec_packet_trimming_buffer")


if __name__ == "__main__":
    unittest.main()
