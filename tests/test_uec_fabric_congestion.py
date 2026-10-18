from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.uec_fabric_congestion.run import (
    DEFAULT_CSV,
    DEFAULT_INPUT,
    DEFAULT_JSON,
    evaluate_fixture,
    evaluate_header_efficiency,
    evaluate_multipath_spraying,
    evaluate_nscc_smartt_swift_reaction,
)


class UecFabricCongestionTest(unittest.TestCase):
    def test_committed_fixture_and_cli_check(self) -> None:
        fixture = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        results = evaluate_fixture(fixture)
        self.assertEqual(results["experiment_id"], "uec_fabric_congestion")
        self.assertEqual(results["evidence_type"], "own-offline")
        self.assertEqual(len(results["header_efficiency"]), 3)
        self.assertEqual(len(results["congestion_control"]), 5)
        self.assertEqual(len(results["multipath_spraying"]), 3)

        proc = subprocess.run(
            [sys.executable, "experiments/uec_fabric_congestion/run.py", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertTrue(DEFAULT_JSON.exists())
        self.assertTrue(DEFAULT_CSV.exists())

    def test_scale_up_ufh_small_header_beats_rocev2(self) -> None:
        res = evaluate_header_efficiency(
            {
                "case_id": "ufh_64b",
                "payload_bytes": 64,
                "line_rate_gbps": 800.0,
                "pds_mode": "rudi",
                "ses_hdr_type": "small_onm",
                "tss_security": False,
                "sync_ext": False,
                "scale_up_ufh": True,
            }
        )
        self.assertEqual(res["wire_bytes_saved_vs_rocev2"], 35)
        self.assertGreater(res["uet_goodput_gbps"], res["rocev2_goodput_gbps"])
        self.assertAlmostEqual(res["goodput_gain_pct_vs_rocev2"], 27.56, places=2)

    def test_nscc_branches_and_quick_adapt(self) -> None:
        noop = evaluate_nscc_smartt_swift_reaction(
            {
                "case_id": "ecn_low_delay",
                "mtu_bytes": 4096,
                "bw_gbps": 400.0,
                "base_rtt_us": 12.0,
                "trim_supp": True,
                "init_cwnd_bdp_fraction": 1.0,
                "rtt_sample_us": 15.0,
                "ecn_marked": True,
                "packet_trimmed_or_nack": False,
                "acked_mtus": 8,
                "achieved_bytes_in_qa_window": 600000,
            }
        )
        self.assertEqual(noop["nscc_branch"], "ecn_low_delay_noop_switch_path")
        self.assertEqual(noop["nscc_next_cwnd_bytes"], 600000)

        qa = evaluate_nscc_smartt_swift_reaction(
            {
                "case_id": "trim_qa",
                "mtu_bytes": 4096,
                "bw_gbps": 400.0,
                "base_rtt_us": 12.0,
                "trim_supp": True,
                "init_cwnd_bdp_fraction": 1.0,
                "rtt_sample_us": 42.0,
                "ecn_marked": True,
                "packet_trimmed_or_nack": True,
                "acked_mtus": 4,
                "achieved_bytes_in_qa_window": 37500,
            }
        )
        self.assertEqual(qa["nscc_branch"], "quick_adapt")
        self.assertEqual(qa["nscc_next_cwnd_bytes"], 37500)
        self.assertLess(qa["nscc_next_cwnd_bytes"], qa["swift_next_cwnd_bytes"])

    def test_reps_outperforms_oblivious_under_hotspot_congestion(self) -> None:
        spray = evaluate_multipath_spraying(
            {
                "case_id": "hotspot",
                "num_paths": 16,
                "congested_paths": 4,
                "failed_blackhole_paths": 0,
                "total_packets": 1024,
                "mtu_bytes": 4096,
                "healthy_path_bw_gbps": 50.0,
                "congested_path_bw_gbps": 10.0,
                "rto_penalty_us": 1000.0,
            }
        )
        self.assertEqual(spray["reps_congested_pkts"], 4)
        self.assertEqual(spray["oblivious_congested_pkts"], 256)
        self.assertGreater(spray["reps_effective_bw_gbps"], spray["selective_effective_bw_gbps"])
        self.assertGreater(spray["selective_effective_bw_gbps"], spray["oblivious_effective_bw_gbps"])

    def test_invalid_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_header_efficiency(
                {
                    "case_id": "bad_pds",
                    "payload_bytes": 64,
                    "line_rate_gbps": 400.0,
                    "pds_mode": "invalid",
                }
            )
        with self.assertRaises(ValueError):
            evaluate_multipath_spraying(
                {
                    "case_id": "non_pow2",
                    "num_paths": 12,
                    "congested_paths": 2,
                    "total_packets": 100,
                }
            )


if __name__ == "__main__":
    unittest.main()
