from fractions import Fraction
import json
import statistics
import subprocess
import unittest

from experiments.affine_scan_contract_selection import run as m
from experiments.affine_scan_contract_selection.render import JS


class AffineScanTests(unittest.TestCase):
    def test_exact_scan_and_independent_expansion(self):
        for n in (1,2,3,7,13):
            values=[(Fraction((i%3)+1,3),Fraction(i-2,7)) for i in range(n)]
            for initial in (0,7,Fraction(-2,3)):
                expected=m.expanded_oracle(values,initial)
                self.assertEqual(m.serial(values,initial),expected)
                self.assertEqual(m.apply(m.prefix_scan(values),initial),expected)

    def test_associative_not_commutative(self):
        a,b,c=(2,1),(3,4),(4,2)
        self.assertEqual(m.compose(m.compose(a,b),c),m.compose(a,m.compose(b,c)))
        self.assertNotEqual(m.compose(a,b),m.compose(b,a))
        self.assertEqual(m.compose(a,(1,0)),a)
        self.assertEqual(m.compose((1,0),a),a)

    def test_reset_erases_prefix_dependence(self):
        values=[(2,1),(3,4),(4,2),(5,3)]
        reset=m.zero_resets(values,{2})
        a=m.serial(reset,7);b=m.serial(reset,-100)
        self.assertNotEqual(a[:2],b[:2])
        self.assertEqual(a[2:],b[2:])
        self.assertEqual(a,[15,49,2,13])
        self.assertEqual(m.apply(m.prefix_scan(reset),7),a)

    def test_padding_identity_and_zero_pair(self):
        values=[(2,1),(3,4)]
        self.assertEqual(m.serial([*values,(1,0),(1,0)],7),[15,49,49,49])
        self.assertEqual(m.serial([*values,(0,0)],7)[-1],0)

    def test_chunk_carry_and_wrong_restart(self):
        values=[(2,1),(3,4),(4,2),(5,3)]
        for chunk in (1,2,3,4,8):
            self.assertEqual(m.chunked(values,chunk,7),m.serial(values,7))
        self.assertNotEqual(m.chunked(values,2,7,True),m.serial(values,7))
        self.assertEqual(m.chunked(values,2,7,True)[-1],153)

    def test_invalid_inputs_and_budget(self):
        with self.assertRaises(ValueError):
            m.prefix_scan([])
        with self.assertRaises(ValueError):
            m.zero_resets([(1,2)],{1})
        with self.assertRaises(ValueError):
            m.chunked([(1,2)],0)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"length":2**53})

    def test_node_states_resets_and_budget(self):
        values=[[2,1],[3,4],[4,2],[5,3]]
        for reset in (-1,0,2):
            source=JS+"\nconsole.log(JSON.stringify({result:simulate("+json.dumps(values)+f",7,{reset})"+",budget:budget("+json.dumps(m.DEFAULT)+")}))"
            out=json.loads(subprocess.check_output(["node","-e",source],text=True))
            selected=m.zero_resets(values,{reset}) if reset>=0 else values
            expected=[int(v) for v in m.serial(selected,7)]
            self.assertEqual(out["result"]["serial"],expected)
            self.assertEqual(out["result"]["scan"],expected)
            self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_float_timing_paths(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("sequential","tree"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.float_paths(8)
        for x,y in zip(a(),b()):
            self.assertAlmostEqual(x,y,places=12)


if __name__=="__main__":
    unittest.main()
