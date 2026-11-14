from fractions import Fraction
import json
import random
import statistics
import subprocess
import unittest

from experiments.finite_float_cast_policy_selection import run as m
from experiments.finite_float_cast_policy_selection.render import JS,cards


class FiniteFloatPolicyTests(unittest.TestCase):
    def test_independent_piecewise_codebook_and_all_raw_signs(self):
        expected={
            "FLOAT4E2M1":[0,8,16,24,32,48,64,96],
            "FLOAT6E2M3":[2*i for i in range(16)]+[4*(i-8) for i in range(16,24)]+[8*(i-16) for i in range(24,32)],
            "FLOAT6E3M2":list(range(8))+[2*(i-4) for i in range(8,12)]+[4*(i-8) for i in range(12,16)]+
                          [8*(i-12) for i in range(16,20)]+[16*(i-16) for i in range(20,24)]+
                          [32*(i-20) for i in range(24,28)]+[64*(i-24) for i in range(28,32)],
        }
        for fmt,ticks in expected.items():
            self.assertEqual(m.book(fmt)["positive_ticks"],ticks)
            for raw in range(2*len(ticks)):
                result=m.decode(raw,fmt)
                self.assertEqual(result["magnitude_ticks"],ticks[raw%len(ticks)])
                self.assertEqual(result["negative"],raw>=len(ticks))
                self.assertEqual(result["negative_zero"],raw==len(ticks))

    def test_all_adjacent_midpoint_rne_ties(self):
        for fmt in m.FORMATS:
            ticks=m.book(fmt)["positive_ticks"]
            for code in range(len(ticks)-1):
                midpoint=Fraction(ticks[code]+ticks[code+1],32)
                token=str(float(midpoint));expected=code if code%2==0 else code+1
                self.assertEqual(m.convert(token,fmt)["raw_code"],expected)
                self.assertEqual(m.convert("-"+token,fmt)["raw_code"],expected|m.book(fmt)["sign_bit"])

    def test_nan_explicit_block_and_error_denominator_not_zero(self):
        result=m.evaluate(m.DEFAULT)
        self.assertEqual(result["blocked_nan_rows"],3)
        for row in result["rows"]:
            if row["input"]=="nan":
                self.assertEqual(row["status"],"blocked_nan_policy");self.assertIsNone(row["raw_code"])
                self.assertIsNone(row["absolute_error_ratio"])
        self.assertEqual([v["finite_error_observations"] for v in result["summary"]],[2,2,2])
        empty=m.evaluate({"formats":["FLOAT4E2M1"],"tokens":["nan","+inf"]})
        self.assertIsNone(empty["summary"][0]["mean_absolute_error_ratio"])

    def test_signed_zero_underflow_and_documented_saturation(self):
        for fmt in m.FORMATS:
            sign=m.book(fmt)["sign_bit"]
            self.assertEqual(m.convert("-0",fmt)["raw_code"],sign)
            self.assertTrue(m.convert("-0.0000001",fmt)["negative_zero"])
            self.assertEqual(m.convert("+inf",fmt)["value"],m.book(fmt)["maximum"])
            self.assertEqual(m.convert("-inf",fmt)["value"],"-"+m.book(fmt)["maximum"])
        self.assertEqual(m.convert("25","FLOAT6E3M2")["value"],"24")
        self.assertEqual(m.convert("25","FLOAT6E2M3")["value"],"7.5")
        self.assertEqual(m.convert("5","FLOAT4E2M1")["value"],"4")

    def test_nearest_matches_exhaustive_fraction_oracle(self):
        rng=random.Random(148)
        for fmt in m.FORMATS:
            ticks=m.book(fmt)["positive_ticks"]
            for _ in range(64):
                value=Fraction(rng.randrange(0,1000),32)
                expected=min(range(len(ticks)),key=lambda code:(abs(Fraction(ticks[code],16)-value),code&1,code))
                self.assertEqual(m.nearest(value*16,ticks),expected)

    def test_strict_tokens_types_and_selected_odd_payload(self):
        for token in ("garbage","1/3"," 1","1e100","1000001",None,True):
            with self.assertRaises(ValueError):
                m.convert(token,"FLOAT4E2M1")
        with self.assertRaises(ValueError):
            m.evaluate({"formats":["FLOAT4E2M1","FLOAT4E2M1"],"tokens":["1"]})
        self.assertEqual(m.payload(9)["selected_raw_dtype_payload_bytes"],{"FLOAT4E2M1":5,"FLOAT6E2M3":7,"FLOAT6E3M2":7})

    def test_node_python_exact_rational_signed_zero_policy_parity(self):
        p={"formats":list(m.FORMATS),"tokens":["0","-0","0.25","-0.25","0.75","1.25","2.5","5","10","25",
                                            "1.25000001","-0.0000001","NaN","+inf","-inf"]}
        result=subprocess.run(["node"],input=JS+"\nconsole.log(JSON.stringify(evaluate("+json.dumps(p)+")));",capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),m.evaluate(p))
        result=subprocess.run(["node"],input=JS+"\nconsole.log(JSON.stringify(Object.keys(specs).map(fmt=>Array.from({length:book(fmt).raw_patterns},(_,raw)=>decode(raw,fmt)))));",capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[[m.decode(raw,fmt) for raw in range(m.book(fmt)["raw_patterns"])] for fmt in m.FORMATS])

    def test_sealed_cpu_and_full_primary_matrix(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency));self.assertIsNone(report["native_nan_cast_result"])
        self.assertEqual(cards(report).split('<section class="card">')[3].count("<tr>"),len(report["case"]["outputs"]["rows"])+1)
        for row in latency["rows"]:
            for name in ("bisect","scan"):
                samples=row[name+"_samples_ms"];self.assertEqual(len(samples),7)
                self.assertTrue(all(v>0 for v in samples));self.assertEqual(row[name+"_median_ms"],statistics.median(samples))


if __name__=="__main__":
    unittest.main()
