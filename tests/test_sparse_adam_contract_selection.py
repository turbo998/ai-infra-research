from fractions import Fraction
import json
import math
import random
import subprocess
import unittest

from experiments.sparse_adam_contract_selection import run as m
from experiments.sparse_adam_contract_selection.render import JS


class SparseAdamTests(unittest.TestCase):
    def test_duplicates_before_nonlinear_moment(self):
        table=m.ScalarTable([1,2]);table.step([(0,1),(0,2)])
        self.assertAlmostEqual(table.m[0],1.5)
        self.assertAlmostEqual(table.v[0],4.5)
        self.assertAlmostEqual(1-table.weights[0],m.decimal_delta([3],1),places=14)
        self.assertNotAlmostEqual(1-table.weights[0],.1*3/math.sqrt(5))

    def test_independent_fraction_coalescing(self):
        rng=random.Random(120)
        entries=[(rng.randrange(8),rng.randrange(-5,6)) for _ in range(150)]
        expected={}
        for row,value in entries:
            expected[row]=expected.get(row,Fraction(0))+Fraction(value)
        self.assertEqual(m.coalesce(entries,8),expected)
        self.assertEqual(m.sorted_coalesce(entries,8),expected)
        self.assertEqual(m.coalesce([(1,1),(1,-1)],2),{1:0})

    def test_absent_row_versus_materialized_zero(self):
        absent=m.ScalarTable([1,2]);zero=m.ScalarTable([1,2])
        for table in (absent,zero):
            table.step([(1,4)])
        snapshot=absent.snapshot();absent.step([]);zero.step([(1,0)])
        self.assertEqual(absent.weights,snapshot["weights"])
        self.assertEqual(absent.m,snapshot["first_moments"])
        self.assertEqual(absent.v,snapshot["second_moments"])
        self.assertLess(zero.weights[1],absent.weights[1])
        self.assertAlmostEqual(zero.m[1],1)
        self.assertAlmostEqual(zero.v[1],4)
        self.assertAlmostEqual(absent.weights[1]-zero.weights[1],m.decimal_delta([4,0],2),places=14)

    def test_parameter_time_none_and_empty(self):
        table=m.ScalarTable([1,2]);table.step(None)
        self.assertEqual(table.step_number,0)
        table.step([]);self.assertEqual(table.step_number,1)
        table.step([(1,4)])
        self.assertAlmostEqual(2-table.weights[1],m.decimal_delta([4],2),places=14)
        self.assertNotAlmostEqual(2-table.weights[1],.1)

    def test_epsilon_location_and_multi_step_decimal(self):
        table=m.ScalarTable([1],eps=.2)
        for g in (3,-2,0):
            table.step([(0,g)])
        expected=1-sum(m.decimal_delta([3,-2,0][:t],t,eps=".2") for t in (1,2,3))
        self.assertAlmostEqual(table.weights[0],expected,places=14)
        alternate=.1*3/(3+.2)
        first=m.ScalarTable([1],eps=.2);first.step([(0,3)])
        self.assertNotAlmostEqual(1-first.weights[0],alternate)

    def test_explicit_invalid_and_empty_budget(self):
        for entries in ([(-1,1)],[(2,1)],[(True,1)],[(0,float("nan"))]):
            with self.assertRaises(ValueError):
                m.coalesce(entries,2)
        with self.assertRaises(ValueError):
            m.ScalarTable([1]).step([(0,0)])
        for p in ({**m.DEFAULT,"unique_rows":129},{**m.DEFAULT,"rows":0},{**m.DEFAULT,"rows":2**53-1}):
            with self.assertRaises(ValueError):
                m.estimate_budget(p)
        out=m.estimate_budget({**m.DEFAULT,"stored_rows":0,"unique_rows":0})
        self.assertEqual(out["selected_raw_sparse_gradient_bytes"],0)
        self.assertGreater(out["selected_dense_float32_two_moments_bytes"],0)

    def test_node_first_update_and_budget_parity(self):
        cases=[([(0,1),(0,2)],1,0),([(1,4)],2,1e-8),([(0,-2),(0,1)],3,.2)]
        source="const DEFAULT="+json.dumps(m.DEFAULT)+";\n"+JS+"\nconsole.log(JSON.stringify({cases:"+json.dumps(cases)+".map(([e,t,p])=>calculate(e,t,p)),budget:estimate(DEFAULT)}));"
        result=json.loads(subprocess.check_output(["node","-e",source],text=True))
        for values,(entries,t,eps) in zip(result["cases"],cases):
            table=m.ScalarTable([1,2],eps=eps);table.step_number=t-1;table.step(entries)
            for value in values:
                self.assertAlmostEqual(value["weight_after"],table.weights[value["row"]],places=14)
        self.assertEqual(result["budget"],m.estimate_budget(m.DEFAULT))

    def test_sealed_report_and_timing(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("hash","sorted"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],sorted(samples)[3])
        self.assertFalse(report["optimizer_state_compaction_claim"])


if __name__=="__main__":
    unittest.main()
