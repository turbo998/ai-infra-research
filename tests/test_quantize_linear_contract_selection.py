from fractions import Fraction
import json
import math
import subprocess
import unittest

from experiments.quantize_linear_contract_selection import run as m
from experiments.quantize_linear_contract_selection.render import JS


class QuantizeLinearTests(unittest.TestCase):
    def test_independent_nearest_neighbors_oracle(self):
        for denominator in range(1,10):
            for numerator in range(-40,41):
                value=Fraction(numerator,denominator);low=math.floor(value);high=low+1
                distances=(value-low,high-value)
                expected=low if distances[0]<distances[1] else high if distances[1]<distances[0] else low if low%2==0 else high
                self.assertEqual(m.nearest_even(value),expected)

    def test_zero_point_noncommutativity_and_tie_neighbors(self):
        self.assertEqual(m.quantized_ratio(Fraction(1,2),1),1)
        self.assertEqual(m.nearest_even(Fraction(1,2)+1),2)
        tiny=Fraction(1,2**30)
        self.assertEqual(m.nearest_even(Fraction(1,2)-tiny),0)
        self.assertEqual(m.nearest_even(Fraction(1,2)+tiny),1)
        self.assertEqual(m.nearest_even(-Fraction(1,2)-tiny),-1)

    def test_axis_negative_axis_and_shape_valid_wrong_axis(self):
        matrix=[[Fraction(1,2),Fraction(3,2)],[-Fraction(1,2),-Fraction(3,2)]]
        scales=[Fraction(1),Fraction(1,2)];zeros=[1,-1]
        expected=[[1,2],[1,-4]]
        self.assertEqual(m.quantize(matrix,scales,zeros),expected)
        self.assertEqual(m.quantize(matrix,scales,zeros,-1),expected)
        self.assertNotEqual(m.quantize(matrix,scales,zeros,0),expected)

    def test_saturation_and_unsigned_domain(self):
        self.assertEqual(m.quantized_ratio(130,1),127)
        self.assertEqual(m.quantized_ratio(-140,1),-128)
        self.assertEqual(m.quantized_ratio(-3,0,"uint8"),0)
        self.assertEqual(m.quantized_ratio(300,0,"uint8"),255)
        with self.assertRaises(ValueError):
            m.quantized_ratio(1,-1,"uint8")

    def test_reconstruction_error_and_exact_reciprocal(self):
        values=[Fraction(n,4) for n in range(-16,17)]
        scales=[Fraction(1,2),Fraction(1,4),Fraction(3,2),Fraction(1)]
        zeros=[-3,1,7,0]
        result=m.flat_quantize(values,scales,zeros,True)
        self.assertEqual(result,m.flat_quantize(values,scales,zeros,False))
        for i,(value,q) in enumerate(zip(values,result)):
            self.assertLessEqual(abs(value-(q-zeros[i%4])*scales[i%4]),scales[i%4]/2)

    def test_invalid_inputs_and_budget(self):
        for args in (([[1]],[0],[0],1),([[1,2]],[1],[0],1),([[1]],[1],[0],2),([[1],[]],[1],[0],0)):
            with self.assertRaises(ValueError):
                m.quantize(*args)
        for value in (1.0,True):
            with self.assertRaises(ValueError):
                m.nearest_even(value)
        with self.assertRaises(ValueError):
            m.quantized_ratio(2**32,0)
        with self.assertRaises(ValueError):
            m.estimate_budget({**m.DEFAULT,"rows":2**53-1})

    def test_node_bigint_matrix_and_budget_parity(self):
        cases=[([["1/2","3/2"],["-1/2","-3/2"]],["1","1/2"],[1,-1],1,"int8"),
               ([["1/2","3/2"],["-1/2","-3/2"]],["1","1/2"],[1,-1],0,"int8"),
               ([["-1","300"]],["1","1"],[0,0],-1,"uint8")]
        source=JS+"\nconsole.log(JSON.stringify({cases:"+json.dumps(cases)+".map(c=>calculate(...c)),budget:estimate("+json.dumps(m.DEFAULT)+")}));"
        result=json.loads(subprocess.check_output(["node","-e",source],text=True))
        for actual,(matrix,scales,zeros,axis,dtype) in zip(result["cases"],cases):
            self.assertEqual(actual,m.quantize([[Fraction(v) for v in row] for row in matrix],[Fraction(v) for v in scales],zeros,axis,dtype))
        self.assertEqual(result["budget"],m.estimate_budget(m.DEFAULT))

    def test_sealed_report_and_timing_medians(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("cached","division"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],sorted(samples)[3])
        self.assertFalse(report["native_precision_compatibility"])


if __name__=="__main__":
    unittest.main()
