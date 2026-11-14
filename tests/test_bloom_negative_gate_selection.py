from fractions import Fraction
from itertools import product
import json
import statistics
import subprocess
import unittest

from experiments.bloom_negative_gate_selection import run as m
from experiments.bloom_negative_gate_selection.render import JS


class BloomNegativeTests(unittest.TestCase):
    def test_explicit_coordinate_universe_independent_oracle(self):
        inserted=[(1,5),(5,9)];bits=m.make_bits(inserted,16)
        for coordinates in product(range(16),repeat=2):
            self.assertEqual(m.contains(bits,coordinates),m.independent_mask_oracle(inserted,coordinates))
        absent=[p for p in product(range(16),repeat=2) if p not in inserted]
        self.assertEqual(sum(m.contains(bits,p) for p in absent),7)
        self.assertEqual(len(absent),254)

    def test_insert_monotonic_and_positive_not_identity(self):
        bits=m.make_bits([(1,5),(5,9)],16)
        self.assertTrue(m.contains(bits,[1,9]))
        for key in ((1,5),(5,9)):
            self.assertTrue(m.contains(bits,key))
        before=m.make_bits([(1,5)],16)
        self.assertTrue(all(not v or bits[i] for i,v in enumerate(before)))

    def test_naive_clear_false_negative_and_counting_trusted_delete(self):
        inserted=[(1,5),(5,9)];bits=m.make_bits(inserted,16)
        for i in inserted[0]:
            bits[i]=False
        self.assertFalse(m.contains(bits,inserted[1]))
        counts=m.counting(inserted,16,[1,5])
        self.assertTrue(all(counts[i]>0 for i in inserted[1]))

    def test_false_positive_counting_delete_is_not_safe(self):
        counts=m.counting([(1,5),(5,9)],16,[1,9])
        self.assertFalse(all(counts[i]>0 for i in (1,5)))
        self.assertFalse(all(counts[i]>0 for i in (5,9)))
        with self.assertRaises(ValueError):
            m.counting([(1,1)],16,[1,1,1])

    def test_layer_bound_without_independence_assumption(self):
        bounds=m.layer_bounds(10000,3)
        self.assertEqual(sum(map(Fraction,bounds["selected_layer_error_limits"])),Fraction(7,800))
        self.assertEqual(Fraction(bounds["finite_union_bound"]),Fraction(7,800))
        self.assertEqual(Fraction(bounds["constant_layer_union_bound"]),Fraction(3,100))

    def test_invalid_coordinates_and_sizing(self):
        for row in ([],[-1],[16],[True]):
            with self.assertRaises(ValueError):
                m.make_bits([row],16)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"layers":17})
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"expected_items":2**53})

    def test_node_finite_universe_and_sizing_parity(self):
        source=JS+"\nconsole.log(JSON.stringify({result:simulate([[1,5],[5,9]],16),budget:budget("+json.dumps(m.DEFAULT)+")}))"
        out=json.loads(subprocess.check_output(["node","-e",source],text=True))
        self.assertEqual(out["result"]["finite_false_positive_count"],7)
        self.assertEqual(out["result"]["finite_absent_pair_count"],254)
        self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_timing_paths(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("bitset","bool"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(64);self.assertEqual(a(),b())
        self.assertIsNone(report["production_false_positive_rate"])


if __name__=="__main__":
    unittest.main()
