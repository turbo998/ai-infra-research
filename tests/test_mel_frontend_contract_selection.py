from fractions import Fraction
import itertools
import json
import math
import subprocess
import unittest

from experiments.mel_frontend_contract_selection import run,render


class MelFrontendTests(unittest.TestCase):
    def test_triangle_matches_independent_clipped_rational_hat(self):
        for left,peak,right in ((0,3,7),(2,5,12),(0,100,300)):
            for value in range(left-2,right+3):
                reference=max(Fraction(0),min(Fraction(value-left,peak-left),Fraction(right-value,right-peak)))
                self.assertAlmostEqual(run.triangle(value,left,peak,right),float(reference),places=14)

    def test_scales_roundtrip_and_piecewise_boundary(self):
        for scale,hz in itertools.product(("htk","slaney"),(0,20,999,1000,1001,8000)):
            self.assertAlmostEqual(run.hertz(run.mel_frequency(hz,scale),scale),hz,places=9)
        self.assertAlmostEqual(run.mel_frequency(1000,"slaney"),15)
        self.assertNotAlmostEqual(run.mel_frequency(1000,"slaney"),run.mel_frequency(1000,"htk"))

    def test_cast_before_normalization_changes_actual_bits(self):
        before,after=run.filters(),run.filters(cast_before=False)
        differences=[(a,b) for row_a,row_b in zip(before,after) for a,b in zip(row_a,row_b) if a!=b]
        self.assertTrue(differences)
        self.assertTrue(all(run.f32(a)==a and run.f32(b)==b for a,b in differences))
        self.assertEqual(len(before),40);self.assertEqual(len(before[0]),257)
        self.assertTrue(all(v>=0 and math.isfinite(v) for row in before for v in row))
        self.assertNotEqual(before,run.filters(scale="htk"))

    def test_dense_sparse_and_permuted_frame_projection(self):
        bank=run.filters()
        frames=[[(i+j+1)/11 for i in range(257)] for j in range(3)]
        dense=run.project(bank,frames)
        self.assertEqual(dense,run.project_sparse(run.compress(bank),frames))
        order=[2,0,1]
        shuffled=run.project(bank,[frames[i] for i in order])
        for position,original in enumerate(order):
            self.assertEqual(shuffled[position],dense[original])
        with self.assertRaises(ValueError):
            run.project(list(zip(*bank)),frames)

    def test_frame_formula_matches_explicit_frame_start_enumeration(self):
        for length,frame,hop,center in itertools.product(range(2,25),range(2,9),range(1,6),(0,1)):
            if not center and length<frame:
                continue
            p={**run.DEFAULT,"samples":length,"frame_length":frame,"hop_length":hop,"center":center,"fft_length":16}
            padded=length+2*(frame//2)*center
            expected=len([start for start in range(padded+1) if start%hop==0 and start+frame<=padded])
            self.assertEqual(run.estimate_budget(p)["frame_count"],expected)

    def test_power_db_is_not_amplitude_db(self):
        self.assertAlmostEqual(run.power_db(4),6.020599913279624)
        self.assertAlmostEqual(20*math.log10(4),2*run.power_db(4))
        for value in (0,-1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):
                run.power_db(value)

    def test_budget_javascript_parity_and_rejections(self):
        profiles=[run.DEFAULT,{**run.DEFAULT,"frame_length":401},{**run.DEFAULT,"center":0},
                  {**run.DEFAULT,"samples":2,"frame_length":2,"fft_length":2,"element_bytes":8}]
        bad=[{"center":2},{"fft_length":511},{"element_bytes":3},{"hop_length":0},
             {"samples":1},{"samples":200,"center":0},{"batch":2**53-1}]
        script="const DEFAULT="+json.dumps(run.DEFAULT)+";\n"+render.JS
        script+="console.log(JSON.stringify("+json.dumps(profiles)+".map(estimate)));"
        script+="for(const patch of "+json.dumps(bad)+"){let ok=false;try{estimate({...DEFAULT,...patch})}catch(e){ok=true}if(!ok)throw Error('invalid accepted')}"
        result=subprocess.run(["node","-e",script],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout),[run.estimate_budget(p) for p in profiles])
        for patch in bad:
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,**patch})
        with self.assertRaises(ValueError):
            run.filters(fft_length=3)

    def test_sealed_report_and_deck(self):
        report=json.loads((run.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,run.build_report(report["latency"]))
        cards=render.cards(report)
        self.assertEqual(cards.count('<section class="card">'),7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",cards)
        self.assertNotIn("<script",cards)
        self.assertEqual(cards.count("<table>"),cards.count("</table>"))
        self.assertIsNone(report["native_filter_bit_exactness"])


if __name__=="__main__":
    unittest.main()
