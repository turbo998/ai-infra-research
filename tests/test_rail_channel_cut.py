"""Independent unit tests for multi-channel rail and intra-node cut arithmetic."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.rail_channel_cut.run import calculate, main, rail_channel_cut


class RailChannelCutTests(unittest.TestCase):
    def base_case(self, **overrides):
        case = {
            "case_id": "rail_aligned_4ch_crossnic0",
            "node_by_rank": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "message_bytes": 2**30,
            "nic_bytes_per_second": 50 * 10**9,
            "intra_node_bytes_per_second": 400 * 10**9,
            "channels": [
                {
                    "ring": [0, 1, 2, 3, 4, 5, 6, 7],
                    "send_nic_by_node": {"A": "NIC0", "B": "NIC0"},
                    "recv_nic_by_node": {"A": "NIC0", "B": "NIC0"},
                },
                {
                    "ring": [1, 2, 3, 0, 5, 6, 7, 4],
                    "send_nic_by_node": {"A": "NIC1", "B": "NIC1"},
                    "recv_nic_by_node": {"A": "NIC1", "B": "NIC1"},
                },
                {
                    "ring": [2, 3, 0, 1, 6, 7, 4, 5],
                    "send_nic_by_node": {"A": "NIC2", "B": "NIC2"},
                    "recv_nic_by_node": {"A": "NIC2", "B": "NIC2"},
                },
                {
                    "ring": [3, 0, 1, 2, 7, 4, 5, 6],
                    "send_nic_by_node": {"A": "NIC3", "B": "NIC3"},
                    "recv_nic_by_node": {"A": "NIC3", "B": "NIC3"},
                },
            ],
        }
        case.update(overrides)
        return rail_channel_cut(**case)

    def test_rail_aligned_splits_per_nic_payload_by_four(self):
        result = self.base_case()
        self.assertEqual(result["payload_per_channel_edge_bytes"], 469762048)
        self.assertEqual(result["cross_nic_node_channel_assignments"], 0)
        self.assertEqual(result["max_nic_direction_bytes"], 469762048)
        self.assertEqual(result["nic_payload_time_lower_bound_seconds"], 0.00939524096)
        self.assertEqual(result["dominant_bottleneck"], "nic")

    def test_cross_rail_skew_doubles_bottleneck_nic_time(self):
        channels = [
            {
                "ring": [0, 1, 2, 3, 4, 5, 6, 7],
                "send_nic_by_node": {"A": "NIC0", "B": "NIC0"},
                "recv_nic_by_node": {"A": "NIC0", "B": "NIC0"},
            },
            {
                "ring": [1, 2, 3, 0, 5, 6, 7, 4],
                "send_nic_by_node": {"A": "NIC0", "B": "NIC1"},
                "recv_nic_by_node": {"A": "NIC1", "B": "NIC0"},
            },
            {
                "ring": [2, 3, 0, 1, 6, 7, 4, 5],
                "send_nic_by_node": {"A": "NIC2", "B": "NIC2"},
                "recv_nic_by_node": {"A": "NIC2", "B": "NIC2"},
            },
            {
                "ring": [3, 0, 1, 2, 7, 4, 5, 6],
                "send_nic_by_node": {"A": "NIC3", "B": "NIC3"},
                "recv_nic_by_node": {"A": "NIC3", "B": "NIC3"},
            },
        ]
        result = self.base_case(case_id="skew", channels=channels)
        self.assertEqual(result["cross_nic_node_channel_assignments"], 2)
        self.assertEqual(result["max_nic_direction_bytes"], 939524096)
        self.assertEqual(result["nic_payload_time_lower_bound_seconds"], 0.01879048192)

    def test_intra_node_phb_bottleneck_overrides_four_rail_nics(self):
        channels = [
            {
                "ring": [0, 1, 2, 3, 4, 5, 6, 7],
                "send_nic_by_node": {"A": f"NIC{i}", "B": f"NIC{i}"},
                "recv_nic_by_node": {"A": f"NIC{i}", "B": f"NIC{i}"},
            }
            for i in range(4)
        ]
        result = self.base_case(
            case_id="intra_phb",
            channels=channels,
            intra_node_bytes_per_second=32 * 10**9,
        )
        self.assertEqual(result["dominant_bottleneck"], "intra_node")
        self.assertEqual(result["intra_node_payload_time_lower_bound_seconds"], 0.058720256)
        self.assertEqual(result["combined_payload_time_lower_bound_seconds"], 0.058720256)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.base_case(case_id="bad id")
        with self.assertRaises(ValueError):
            self.base_case(channels=[])
        with self.assertRaises(ValueError):
            self.base_case(message_bytes=2**30 + 1)
        for bad in ({}, {"schema_version": 1, "synthetic": False, "cases": [{}]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                calculate(bad)

    def test_cli_roundtrip_and_check(self):
        fixture = json.loads(
            (Path(__file__).resolve().parents[1] / "experiments" / "rail_channel_cut" / "input.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.json").write_text(json.dumps(fixture), encoding="utf-8")
            with patch("experiments.rail_channel_cut.run.HERE", root), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--check"]), 1)
                self.assertEqual(main([]), 0)
                self.assertEqual(main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
