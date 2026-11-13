from fractions import Fraction
import json
import math
import subprocess
import unittest

from experiments.kernel_cholesky_contract_selection import run as m
from experiments.kernel_cholesky_contract_selection.render import JS


class KernelCholeskyTests(unittest.TestCase):
    def test_rational_solve_and_factor_reconstruction(self):
        matrix=m.shifted([[1,1],[1,1]],Fraction(1,10))
        l=m.factor(matrix);x=m.solve_factor(l,[1,1])
        expected=m.rational_solve(matrix,[1,1])
        for actual,value in zip(x,expected):
            self.assertAlmostEqual(actual,float(value),places=14)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(sum(l[i][k]*l[j][k] for k in range(2)),float(matrix[i][j]),places=14)

    def test_independent_elimination_multiple_matrices(self):
        for n in range(2,7):
            matrix=[[Fraction(1,2)**abs(i-j)+(Fraction(1,4) if i==j else 0) for j in range(n)] for i in range(n)]
            rhs=[Fraction(i+1,n) for i in range(n)]
            x=m.solve_factor(m.factor(matrix),rhs)
            for actual,value in zip(x,m.rational_solve(matrix,rhs)):
                self.assertAlmostEqual(actual,float(value),places=12)
            self.assertLess(m.residual(matrix,x,rhs),1e-12)

    def test_singular_asymmetric_indefinite_rejection(self):
        for matrix in ([[1,1],[1,1]],[[1,2],[2,1]],[[1,2],[1,3]],[[1,float("nan")],[float("nan"),1]],[]):
            with self.assertRaises(ValueError):
                m.factor(matrix)
        with self.assertRaises(ValueError):
            m.rational_solve([[1,1],[1,1]],[1,1])

    def test_mean_latent_noisy_and_logdet(self):
        matrix=m.shifted([[1,1],[1,1]],Fraction(1,10));l=m.factor(matrix)
        x=m.rational_solve(matrix,[1,1])
        self.assertEqual(sum(x),Fraction(20,21))
        self.assertEqual(1-sum(x),Fraction(1,21))
        self.assertEqual(1-sum(x)+Fraction(1,10),Fraction(31,210))
        self.assertAlmostEqual(2*sum(math.log(l[i][i]) for i in range(2)),math.log(.21),places=14)

    def test_stale_factor_and_matrix_permutation(self):
        old=m.shifted([[1,1],[1,1]],Fraction(1,10))
        new=m.shifted([[1,1],[1,1]],Fraction(1,5))
        stale=m.solve_factor(m.factor(old),[1,1])
        self.assertGreater(m.residual(new,stale,[1,1]),.04)
        self.assertLess(m.residual(new,m.solve_factor(m.factor(new),[1,1]),[1,1]),1e-12)
        matrix=[[2,1],[1,3]];rhs=[1,4];order=[1,0]
        actual=m.solve_factor(m.factor([[matrix[i][j] for j in order] for i in order]),[rhs[i] for i in order])
        original=m.solve_factor(m.factor(matrix),rhs)
        for a,i in zip(actual,order):
            self.assertAlmostEqual(a,original[i],places=14)

    def test_invalid_rhs_shift_and_budget(self):
        with self.assertRaises(ValueError):
            m.solve_factor(m.factor([[1]]),[1,2])
        with self.assertRaises(ValueError):
            m.shifted([[1]],-1)
        with self.assertRaises(ValueError):
            m.estimate_budget({**m.DEFAULT,"training_points":2**53-1})

    def test_node_model_budget_and_explicit_zero_shift(self):
        source="const DEFAULT="+json.dumps(m.DEFAULT)+";\n"+JS+"\nconsole.log(JSON.stringify({result:calculate(.1),budget:estimate(DEFAULT)}));"
        result=json.loads(subprocess.check_output(["node","-e",source],text=True))
        expected=m.rational_solve(m.shifted([[1,1],[1,1]],Fraction(1,10)),[1,1])
        for actual,value in zip(result["result"]["solution"],expected):
            self.assertAlmostEqual(actual,float(value),places=14)
        self.assertEqual(result["budget"],m.estimate_budget(m.DEFAULT))
        source="const DEFAULT="+json.dumps(m.DEFAULT)+";\n"+JS+"\ntry{calculate(0);process.exit(1)}catch(e){console.log(e.message)}"
        self.assertTrue(subprocess.check_output(["node","-e",source],text=True).strip())

    def test_sealed_report_and_timing_medians(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("cached","refactor"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],sorted(samples)[3])
        self.assertFalse(report["native_rescue_reproduced"])


if __name__=="__main__":
    unittest.main()
