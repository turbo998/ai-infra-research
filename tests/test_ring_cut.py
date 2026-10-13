"""Independent assertions for original directed Ring cut arithmetic."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.ring_cut.run import calculate, main, ring_cut


class RingCutTests(unittest.TestCase):
    def case(self, **changes):
        inputs = {
            "node_by_rank": ["A"] * 4 + ["B"] * 4,
            "ring": list(range(8)),
            "message_bytes": 2**30,
            "nic_bytes_per_second": 50 * 10**9,
        }
        inputs.update(changes)
        return ring_cut(**inputs)

    def test_grouped_ring_counts_wraparound(self):
        result = self.case()
        self.assertEqual(result["payload_per_directed_edge_bytes"], 1879048192)
        self.assertEqual(result["outgoing_cross_edges_by_node"], {"A": 1, "B": 1})
        self.assertEqual(result["node_network_sent_bytes"], {"A": 1879048192, "B": 1879048192})
        self.assertEqual(result["fabric_network_sent_bytes"], 3758096384)
        self.assertEqual(result["nic_payload_time_lower_bound_seconds"], 0.03758096384)

    def test_alternating_ring_shared_nic(self):
        result = self.case(ring=[0, 4, 1, 5, 2, 6, 3, 7])
        self.assertEqual(result["cross_node_directed_edges"], 8)
        self.assertEqual(result["node_network_sent_bytes"], {"A": 7516192768, "B": 7516192768})
        self.assertEqual(result["fabric_network_sent_bytes"], 15032385536)
        self.assertEqual(result["nic_payload_time_lower_bound_seconds"], 0.15032385536)

    def test_all_local_is_not_zero_collective_time(self):
        result = self.case(node_by_rank=["A"] * 8)
        self.assertEqual(result["cross_node_directed_edges"], 0)
        self.assertEqual(result["nic_payload_time_lower_bound_seconds"], 0)
        self.assertGreater(result["payload_per_directed_edge_bytes"], 0)

    def test_two_rank_ring_both_directions(self):
        result = self.case(node_by_rank=["A", "B"], ring=[1, 0], message_bytes=8)
        self.assertEqual(result["payload_per_directed_edge_bytes"], 8)
        self.assertEqual(result["node_network_sent_bytes"], {"A": 8, "B": 8})

    def test_invalid_rank_maps_and_permutations(self):
        for nodes in ([], ["A"], ["A", ""], ["A", 1], ["A", "node A"]):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                self.case(node_by_rank=nodes)
        for ring in ([0] * 8, list(range(7)), list(range(9)), [False] + list(range(1, 8)), None):
            with self.subTest(ring=ring), self.assertRaises(ValueError):
                self.case(ring=ring)

    def test_invalid_units_and_indivisible_payload(self):
        for field in ("message_bytes", "nic_bytes_per_second"):
            for value in (0, -1, True, 1.5, float("nan"), "8"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.case(**{field: value})
        with self.assertRaises(ValueError):
            self.case(message_bytes=9)

    def test_invalid_fixture_schema(self):
        for value in ({}, {"schema_version": True, "synthetic": True, "cases": [{}]},
                      {"schema_version": 1, "synthetic": True, "cases": []},
                      {"schema_version": 1, "synthetic": False, "cases": [{}]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculate(value)

    def test_nonfinite_derived_time_is_rejected(self):
        with self.assertRaises(ValueError):
            self.case(message_bytes=8 * 10**400, nic_bytes_per_second=1)
        with self.assertRaises(ValueError):
            self.case(nic_bytes_per_second=10**400)

    def test_cli_roundtrip_stale_result_and_invalid_input(self):
        fixture = {
            "schema_version": 1, "synthetic": True, "cases": [{
                "node_by_rank": ["A", "B"], "ring": [0, 1],
                "message_bytes": 8, "nic_bytes_per_second": 10,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.json").write_text(json.dumps(fixture), encoding="utf-8")
            with patch("experiments.ring_cut.run.HERE", root), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--check"]), 1)
                self.assertEqual(main([]), 0)
                self.assertEqual(main(["--check"]), 0)
                (root / "results.json").write_text("{}\n", encoding="utf-8")
                self.assertEqual(main(["--check"]), 1)
                self.assertEqual((root / "results.json").read_text(encoding="utf-8"), "{}\n")
                (root / "input.json").write_text("not-json", encoding="utf-8")
                self.assertEqual(main([]), 1)


if __name__ == "__main__":
    unittest.main()
