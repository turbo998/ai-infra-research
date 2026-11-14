from fractions import Fraction
import json
import statistics
import subprocess
import unittest

from experiments.polyphase_resampling_contract_selection import run as m
from experiments.polyphase_resampling_contract_selection.render import JS


class PolyphaseResamplingTests(unittest.TestCase):
    def test_exact_polyphase_against_dense_convolution(self):
        for length in (1,2,5,8):
            x=[Fraction((i%5)-2,3) for i in range(length)]
            for up,down in ((3,2),(2,3),(4,6),(5,1),(1,5),(2,2)):
                self.assertEqual(m.polyphase(x,m.FIR,up,down),m.dense_oracle(x,m.FIR,up,down))

    def test_gcd_ceiling_and_identity_copy(self):
        self.assertEqual(m.rate(48000,44100),(160,147))
        p=m.budget(m.DEFAULT)
        self.assertEqual(p["output_samples"],48000)
        self.assertEqual(m.budget({**m.DEFAULT,"input_samples":5,"up":3,"down":2})["output_samples"],8)
        x=[1,2,3]
        result=m.polyphase(x,[Fraction(0),Fraction(0),Fraction(0)],2,2)
        self.assertEqual(result,x)
        self.assertIsNot(result,x)

    def test_phase_and_gain_negative_controls(self):
        x=[1,2,3,4,5]
        correct=m.polyphase(x,m.FIR,3,2)
        shifted=m.polyphase(x,m.FIR,3,2,1)
        missing_gain=m.polyphase(x,m.FIR,3,2,scale=False)
        self.assertTrue(all(a!=b for a,b in zip(correct,shifted)))
        self.assertEqual(correct,[v*3 for v in missing_gain])

    def test_nyquist_alias_and_zero_boundary(self):
        h=[Fraction(1,4),Fraction(1,2),Fraction(1,4)]
        x=[1,-1]*8
        filtered=m.polyphase(x,h,1,2)
        self.assertEqual(filtered[0],Fraction(1,4))
        self.assertEqual(filtered[1:],[0]*7)
        self.assertEqual(x[::2],[1]*8)
        dc=m.polyphase([1]*16,h,1,2)
        self.assertEqual(dc,[Fraction(3,4)]+[1]*7)

    def test_impulse_superposition(self):
        x=[1,2,3,4,5];y=[5,4,3,2,1]
        total=m.polyphase([a+b for a,b in zip(x,y)],m.FIR,3,2)
        self.assertEqual(total,[a+b for a,b in zip(m.polyphase(x,m.FIR,3,2),m.polyphase(y,m.FIR,3,2))])
        impulse=[0]*8;impulse[2]=1
        self.assertEqual(m.polyphase(impulse,m.FIR,3,2),m.dense_oracle(impulse,m.FIR,3,2))

    def test_invalid_rate_filter_and_budget(self):
        for up,down in ((0,1),(1,-1),(True,1),(1.2,2)):
            with self.assertRaises(ValueError):
                m.rate(up,down)
        for h in ([1,1],[1,2,4],[]):
            with self.assertRaises(ValueError):
                m.polyphase([1,2],h,3,2)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"input_samples":2**53-1,"up":2,"down":1})

    def test_node_rate_phase_and_budget(self):
        for u,d in ((3,2),(2,3),(6,4),(1,1)):
            x=[1,2,3,4,5]
            source=JS+"\nconsole.log(JSON.stringify({values:simulate("+json.dumps(x)+f",{u},{d})"+",budget:budget("+json.dumps(m.DEFAULT)+")}))"
            out=json.loads(subprocess.check_output(["node","-e",source],text=True))
            self.assertEqual(out["budget"],m.budget(m.DEFAULT))
            for actual,exact in zip(out["values"],m.polyphase(x,m.FIR,u,d)):
                self.assertAlmostEqual(actual,float(exact),places=13)

    def test_sealed_report_and_float_timing_paths(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("polyphase","dense"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.float_paths(8)
        self.assertEqual(a(),b())
        self.assertIsNone(report["audio_quality"])


if __name__=="__main__":
    unittest.main()
