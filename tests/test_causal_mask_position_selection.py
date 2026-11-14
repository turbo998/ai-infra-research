from fractions import Fraction
import json
import statistics
import subprocess
import unittest

from experiments.causal_mask_position_selection import run as m
from experiments.causal_mask_position_selection.render import JS


class CausalMaskTests(unittest.TestCase):
    def test_contiguous_suffix_independent_position_oracle(self):
        for keys in (1,2,5,9):
            values=list(range(1,keys+1))
            for queries in range(1,keys+1):
                qpos=list(range(keys-queries,keys));kpos=list(range(keys))
                mask=m.triangular_mask(queries,keys,True)
                self.assertEqual(mask,m.positional_mask(qpos,kpos))
                self.assertEqual(m.attend_uniform(values,mask),m.independent_prefix_oracle(values,qpos,kpos))

    def test_rectangular_and_square_alignment(self):
        self.assertEqual([sum(row) for row in m.triangular_mask(2,5)],[1,2])
        self.assertEqual([sum(row) for row in m.triangular_mask(2,5,True)],[4,5])
        self.assertEqual(m.triangular_mask(4,4),m.triangular_mask(4,4,True))
        values=[1,2,4,8,16]
        self.assertEqual(m.attend_uniform(values,m.triangular_mask(1,5))[0]["value"],"1")
        self.assertEqual(m.attend_uniform(values,m.triangular_mask(1,5,True))[0]["value"],"31/5")

    def test_boolean_polarity_and_explicit_empty_row(self):
        values=[1,2,4,8,16];mask=m.positional_mask([3,4],list(range(5)))
        inverted=[[not v for v in row] for row in mask]
        output=m.attend_uniform(values,inverted)
        self.assertEqual(output,[{"kind":"value","value":"16"},{"kind":"no_allowed_keys","value":None}])
        self.assertEqual(m.attend_uniform(values,mask)[0]["value"],"15/4")

    def test_gapped_and_permuted_position_axes(self):
        values=[1,2,4,8,16];keys=[0,2,4,6,8];queries=[3,4]
        mask=m.positional_mask(queries,keys)
        self.assertEqual([sum(row) for row in mask],[2,3])
        self.assertNotEqual(mask,m.triangular_mask(2,5,True))
        self.assertEqual(m.attend_uniform(values,mask),m.independent_prefix_oracle(values,queries,keys))
        order=[4,2,0,3,1]
        self.assertEqual(m.attend_uniform([values[i] for i in order],m.positional_mask(queries,[keys[i] for i in order])),m.attend_uniform(values,mask))

    def test_more_queries_than_keys_keeps_empty_rows_tagged(self):
        mask=m.triangular_mask(4,2,True)
        out=m.attend_uniform([1,2],mask)
        self.assertEqual([row["kind"] for row in out],["no_allowed_keys","no_allowed_keys","value","value"])
        self.assertIsNone(out[0]["value"])

    def test_invalid_mask_and_budget(self):
        with self.assertRaises(ValueError):
            m.positional_mask([],[])
        with self.assertRaises(ValueError):
            m.attend_uniform([1,2],[[True]])
        with self.assertRaises(ValueError):
            m.attend_uniform([1],[[1]])
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"keys":2**53})

    def test_node_masks_values_and_payload_parity(self):
        values=[1,2,4,8,16]
        for keys in (list(range(5)),[0,2,4,6,8]):
            source=JS+"\nconsole.log(JSON.stringify({result:simulate("+json.dumps(values)+",[3,4],"+json.dumps(keys)+"),budget:budget("+json.dumps(m.DEFAULT)+")}))"
            out=json.loads(subprocess.check_output(["node","-e",source],text=True))
            expected=m.attend_uniform(values,m.positional_mask([3,4],keys))
            self.assertEqual(out["result"]["actual_allow"],m.positional_mask([3,4],keys))
            for actual,exact in zip(out["result"]["actual_outputs"],expected):
                self.assertAlmostEqual(actual["value"],float(Fraction(exact["value"])),places=12)
            self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_float_paths(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("prefix","dense"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.float_paths(128)
        self.assertEqual(a(),b())
        self.assertIsNone(report["native_all_masked_behavior"])


if __name__=="__main__":
    unittest.main()
