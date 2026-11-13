from fractions import Fraction
import itertools
import json
import subprocess
import unittest

from experiments.conformal_rank_contract_selection import run as m
from experiments.conformal_rank_contract_selection.render import JS


class ConformalRankTests(unittest.TestCase):
    def test_corrected_rank_independent_integer_inequalities(self):
        for n in range(1,25):
            for numerator in range(1,10):
                alpha=Fraction(numerator,10);k=m.rank(n,alpha)
                self.assertGreaterEqual(Fraction(k),(n+1)*(1-alpha))
                self.assertLess(Fraction(k-1),(n+1)*(1-alpha))

    def test_finite_and_unbounded_boundaries(self):
        scores=list(range(1,10))
        self.assertEqual(m.threshold(scores,Fraction(3,20))["value"],"9")
        bound=m.threshold(scores,Fraction(1,20))
        self.assertEqual(bound["kind"],"unbounded")
        self.assertIsNone(bound["value"])
        self.assertTrue(m.contains(bound,10**20))
        self.assertEqual(m.threshold(list(range(19)),Fraction(1,20))["kind"],"finite")

    def test_exhaustive_exchangeable_world_and_naive_control(self):
        world=list(range(1,11));alpha=Fraction(3,20)
        self.assertEqual(m.exhaustive_holdout(world,alpha)["covered"],9)
        self.assertEqual(m.exhaustive_holdout(world,alpha,"naive_n")["covered"],8)
        self.assertEqual(m.exhaustive_holdout(world,Fraction(1,20))["covered"],10)
        self.assertEqual(m.exhaustive_holdout(world,Fraction(1,20),"clip_max")["covered"],9)

    def test_ties_permutation_and_scale(self):
        alpha=Fraction(1,4)
        scores=[1,3,2,2]
        expected=m.threshold(scores,alpha)
        for values in itertools.permutations(scores):
            self.assertEqual(m.threshold(values,alpha),expected)
        self.assertEqual(Fraction(m.threshold([v*3 for v in scores],alpha)["value"]),3*Fraction(expected["value"]))
        self.assertEqual(m.exhaustive_holdout([2]*10,alpha)["covered"],10)
        self.assertEqual(m.exhaustive_holdout([2]*10,alpha,inclusive=False)["covered"],0)

    def test_negative_invalid_and_no_silent_fallback(self):
        for scores,alpha in (([],Fraction(1,10)),([-1],Fraction(1,10)),([1],Fraction(0)),([1],Fraction(1)),([1.0],Fraction(1,10))):
            with self.assertRaises(ValueError):
                m.threshold(scores,alpha)
        with self.assertRaises(ValueError):
            m.exhaustive_holdout([1,2],Fraction(1,10),"invented")
        with self.assertRaises(ValueError):
            m.estimate_budget({**m.DEFAULT,"calibration_count":2**53-1})

    def test_exact_minimum_finite_n(self):
        for denominator in (3,10,20,100):
            p={**m.DEFAULT,"alpha_num":1,"alpha_den":denominator}
            minimum=m.estimate_budget(p)["minimum_n_for_finite_single_threshold"]
            self.assertEqual(minimum,denominator-1)
            self.assertLessEqual(m.rank(minimum,Fraction(1,denominator)),minimum)
            if minimum>1:
                self.assertGreater(m.rank(minimum-1,Fraction(1,denominator)),minimum-1)

    def test_node_rank_budget_and_tagged_boundary_parity(self):
        cases=[(list(range(1,10)),3,20),(list(range(1,10)),1,20),([2]*4,1,4)]
        source="const DEFAULT="+json.dumps(m.DEFAULT)+";\n"+JS+"\nconsole.log(JSON.stringify({cases:"+json.dumps(cases)+".map(c=>calculate(...c)),budget:estimate(DEFAULT)}));"
        result=json.loads(subprocess.check_output(["node","-e",source],text=True))
        for actual,(scores,p,d) in zip(result["cases"],cases):
            self.assertEqual(actual,m.threshold(scores,Fraction(p,d)))
        self.assertEqual(result["budget"],m.estimate_budget(m.DEFAULT))

    def test_sealed_report_and_sample_medians(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("heap","sort"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],sorted(samples)[3])
        self.assertIsNone(report["real_data_coverage"])


if __name__=="__main__":
    unittest.main()
