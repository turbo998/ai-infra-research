from fractions import Fraction
import itertools
import json
import re
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.exponential_histogram_downscale_selection import render, run


class ExponentialHistogramTests(unittest.TestCase):
    def test_exact_boundaries_against_independent_interval_search(self):
        for scale in (0,-1,-2,-3):
            for value in [Fraction(n,d) for n in range(1,33) for d in (1,2,4,8)]:
                index = run.bucket_index(value,scale)
                candidates = [i for i in range(-10,11)
                              if run.bucket_bounds(i,scale)[0]<value<=run.bucket_bounds(i,scale)[1]]
                self.assertEqual(candidates,[index])
        self.assertEqual(run.bucket_index(1),-1)
        self.assertEqual(run.bucket_index(2),0)
        self.assertEqual(run.bucket_index(4),1)
        self.assertEqual(run.bucket_index(Fraction(1,2)),-2)

    def test_signed_downscale_conserves_counts_and_membership(self):
        values = run.fixture()
        fine = run.histogram(values)
        coarse = run.fit(fine,4)
        self.assertEqual((fine.span,coarse.span,coarse.scale),(8,3,-2))
        self.assertEqual(coarse,run.histogram(values,-2))
        self.assertEqual(coarse.count,10)
        self.assertEqual(coarse.zero_count,1)
        r = run.simulate()
        self.assertEqual(r["case"]["wrong_count"],10)
        self.assertEqual(r["case"]["wrong_containment_violations"],3)

    def test_repartition_merge_alignment_empty_identity_and_associativity(self):
        values = run.fixture()
        whole = run.histogram(values,-2)
        for split in range(len(values)+1):
            a,b = run.histogram(values[:split]),run.histogram(values[split:],-1)
            self.assertEqual(run.merge((a,b,run.histogram([])),-2),whole)
        a,b,c = (run.histogram(values[i::3],-i) for i in range(3))
        self.assertEqual(run.merge((run.merge((a,b),-2),c),-2),whole)
        self.assertEqual(run.merge((a,run.merge((b,c),-2)),-2),whole)
        self.assertEqual(run.merge((a,b,c)),whole)
        self.assertEqual(a,run.histogram(values[0::3]))

    def test_quantile_intervals_contain_exact_ranks_not_point_estimates(self):
        for raw in itertools.product((0,Fraction(1,2),1,2,4),repeat=3):
            hist = run.histogram(raw,-1)
            for q in (Fraction(1,3),Fraction(1,2),Fraction(1)):
                self.assertTrue(run.interval_contains(run.quantile_interval(hist,q),run.exact_quantile(raw,q)))
        tail = run.simulate()["quantiles"][-1]
        self.assertEqual((tail["exact"],tail["lower_exclusive"],tail["upper_inclusive"]),("32","16","256"))

    def test_invalid_floats_upscale_bad_state_and_infeasible_span_raise(self):
        for operation in (lambda:run.bucket_index(0),lambda:run.histogram([-1]),lambda:run.histogram([True]),
                          lambda:run.histogram([.5]),lambda:run.histogram([1],1),
                          lambda:run.Histogram(0,((1,1),(1,2))),lambda:run.Histogram(0,((1,0),)),
                          lambda:run.Histogram(0,(),True),lambda:run.downscale(run.histogram([1],-1),0),
                          lambda:run.merge([]),lambda:run.fit(run.histogram([Fraction(1,4),32]),1),
                          lambda:run.quantile_interval(run.histogram([]),Fraction(1,2)),
                          lambda:run.quantile_interval(run.histogram([1]),0)):
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(run.fit(run.histogram([]),1),run.histogram([]))

    def test_selected_circular_capacity_is_not_logical_span_or_retained_memory(self):
        b = run.estimate_budget(run.DEFAULT)
        self.assertEqual((b["positive_fresh_capacity"],b["negative_fresh_capacity"]),(256,32))
        self.assertEqual(b["selected_fresh_counter_bytes_per_state"],2312)
        self.assertEqual(b["selected_live_counter_payload_bytes"],46240000)
        self.assertIsNone(b["native_retained_capacity_bytes"])
        zero = run.estimate_budget({**run.DEFAULT,"positive_span":0,"negative_span":0})
        self.assertEqual(zero["selected_fresh_counter_bytes_per_state"],8)
        for patch in ({"streams":0},{"positive_span":161},{"max_size":True},{"negative_span":-1},
                      {"streams":2**53-1}):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,**patch})

    def test_javascript_exact_capacity_parity(self):
        profiles = [run.DEFAULT,{**run.DEFAULT,"positive_span":0,"negative_span":0},
                    {**run.DEFAULT,"positive_span":128,"negative_span":1},
                    {**run.DEFAULT,"streams":1,"live_states_per_stream":1,"max_size":17,
                     "positive_span":17,"negative_span":16}]
        code = render.CALCULATOR.split("const form=")[0]
        script = code+"\nconst profiles="+json.dumps(profiles)+";\nconsole.log(JSON.stringify(profiles.map(estimateBudget)));"
        script += "for(const bad of [{streams:0},{positive_span:161},{max_size:true},{negative_span:-1},{streams:2**53-1}])"
        script += "{let ok=false;try{estimateBudget({...profiles[0],...bad});}catch(e){ok=true;}if(!ok)throw Error('accepted invalid');}"
        result = subprocess.run(["node","-e",script],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout),[run.estimate_budget(p) for p in profiles])
        self.assertIn('role="alert"',render.workbench(run.simulate()))

    def test_sealed_artifacts_timing_csp_and_solid_bounds(self):
        for path,content in run.outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
        r = run.simulate()
        for row in r["latency"]["rows"]:
            self.assertEqual(row["result_count_error"],0)
            self.assertTrue(row["bucket_state_equal"])
            self.assertEqual(len(row["primary_samples_ms"]),7)
            self.assertEqual(row["primary_median_ms"],statistics.median(row["primary_samples_ms"]))
            self.assertEqual(row["baseline_median_ms"],statistics.median(row["baseline_samples_ms"]))
        deck = render.deck(r)
        self.assertEqual(deck.count('<section class="card"'),7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",deck)
        self.assertNotIn("<script",deck)
        self.assertNotRegex(deck,r'(?:src|href)="https?://')
        svg = render.summary_svg(r)
        self.assertEqual(ET.fromstring(svg).attrib["width"],"1600")
        for points in re.findall(r'<polygon points="([^"]+)',svg):
            for point in points.split():
                self.assertTrue(200<=float(point.split(",")[1])<=620)


if __name__=="__main__":
    unittest.main()
