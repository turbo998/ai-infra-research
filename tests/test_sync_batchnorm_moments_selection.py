import itertools
import json
import math
import re
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.sync_batchnorm_moments_selection import render, run


class SyncBatchNormMomentsTests(unittest.TestCase):
    def test_count_weighted_mean_and_between_mean_variance(self):
        m = run.simulate()["moments"]
        self.assertAlmostEqual(m["mean"],m["exact_reference"]["mean"],places=12)
        self.assertAlmostEqual(m["population_variance"],m["exact_reference"]["population_variance"],places=12)
        self.assertAlmostEqual(m["unbiased_variance"],m["exact_reference"]["unbiased_variance"],places=12)
        self.assertGreater(abs(m["mean_of_rank_means"]-m["mean"]),1)
        self.assertGreater(abs(m["count_weighted_local_variances"]-m["population_variance"]),20)
        self.assertNotEqual(m["mean_of_rank_variances"],m["population_variance"])

    def test_merge_permutations_repartition_and_empty_identity(self):
        shards = run.simulate()["moments"]["shards"]
        values = [x for xs in shards for x in xs]
        reference = run.exact_reference(values)
        for permuted in itertools.permutations(shards):
            s = run.merged([run.summarize(xs) for xs in permuted])
            self.assertAlmostEqual(s.mean,reference["mean"],places=12)
            self.assertAlmostEqual(s.population_variance,reference["population_variance"],places=12)
        for split in range(len(values)+1):
            s = run.merged([run.summarize(values[:split]),run.summarize([]),run.summarize(values[split:])])
            self.assertEqual(s.count,len(values))
            self.assertAlmostEqual(s.population_variance,reference["population_variance"],places=12)

    def test_large_offset_cancellation_against_exact_rationals(self):
        c = run.simulate()["cancellation"]
        self.assertEqual(c["exact_variance"],5)
        self.assertEqual(c["merged_variance"],c["exact_variance"])
        self.assertNotEqual(c["raw_second_moment_variance"],c["exact_variance"])
        centered = [x-c["offset"] for x in c["values"]]
        self.assertEqual(run.summarize(centered).population_variance,c["merged_variance"])

    def test_global_backward_matches_finite_differences_not_local_sums(self):
        g = run.simulate()["gradient"]
        self.assertLess(g["maximum_dx_finite_difference_error"],1e-8)
        self.assertLess(abs(g["analytic"]["dgamma"]-g["finite_difference"]["dgamma"]),1e-8)
        self.assertLess(abs(g["analytic"]["dbeta"]-g["finite_difference"]["dbeta"]),1e-8)
        self.assertAlmostEqual(math.fsum(g["analytic"]["dx"]),0,places=12)
        self.assertGreater(g["maximum_wrong_local_dx_error"],.1)
        with self.assertRaises(ValueError):
            run.gradients([[1,2]],[[1]])

    def test_empty_undefined_estimators_and_explicit_invalid_values(self):
        empty = run.summarize([])
        self.assertIsNone(empty.mean)
        self.assertIsNone(empty.population_variance)
        self.assertIsNone(run.summarize([4]).unbiased_variance)
        one = run.summarize([4])
        self.assertIs(run.merge(empty,one),one)
        with self.assertRaises(ValueError):
            run.forward([])
        with self.assertRaises(ValueError):
            run.summarize([float("nan")])
        with self.assertRaises(ValueError):
            run.forward([1,2],epsilon=0)
        with self.assertRaises(ValueError):
            run.Moments(0,0,0)

    def test_payload_counts_and_two_distinct_representation_guards(self):
        b = run.estimate_budget(run.DEFAULT)
        self.assertEqual(b["forward_packet_bytes_per_rank"],8196)
        self.assertEqual(b["forward_gather_buffer_bytes_per_rank"],32784)
        self.assertEqual(b["backward_reduction_payload_bytes_per_rank"],8192)
        self.assertEqual(b["hypothetical_feature_gather_bytes_per_rank"],4194304)
        odd = run.estimate_budget({**run.DEFAULT,"rank_counts":[2**24+1]})
        even = run.estimate_budget({**run.DEFAULT,"rank_counts":[2**24+2]})
        self.assertFalse(odd["all_packet_counts_exact_in_selected_float"])
        self.assertTrue(even["all_packet_counts_exact_in_selected_float"])
        self.assertTrue(odd["all_saved_rank_counts_fit_int32"])
        wide = run.estimate_budget({**run.DEFAULT,"rank_counts":[2**31],"stat_bytes":8})
        self.assertTrue(wide["all_packet_counts_exact_in_selected_float"])
        self.assertFalse(wide["all_saved_rank_counts_fit_int32"])
        for patch in ({"rank_counts":[]},{"rank_counts":[0,0]},{"rank_counts":[True]},
                      {"channels":1.5},{"stat_bytes":True},{"value_bytes":3},{"channels":2**53}):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,**patch})

    def test_javascript_exact_byte_and_float_count_parity(self):
        profiles = [run.DEFAULT,{**run.DEFAULT,"rank_counts":[0,1,3]},
                    {**run.DEFAULT,"rank_counts":[2**24+1]},
                    {**run.DEFAULT,"rank_counts":[2**24+2]},
                    {**run.DEFAULT,"rank_counts":[2**31],"stat_bytes":8}]
        code = render.CALCULATOR.split("const form=")[0]
        script = code+"\nconst profiles="+json.dumps(profiles)+";\nconsole.log(JSON.stringify(profiles.map(estimateBudget)));"
        script += "for(const bad of [{rank_counts:[]},{rank_counts:[0,0]},{rank_counts:[true]},"
        script += "{channels:1.5},{stat_bytes:true},{value_bytes:3},{channels:2**53}])"
        script += "{let ok=false;try{estimateBudget({...profiles[0],...bad});}catch(e){ok=true;}if(!ok)throw Error('accepted invalid');}"
        result = subprocess.run(["node","-e",script],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout),[run.estimate_budget(p) for p in profiles])
        self.assertIn('role="alert"',render.workbench(run.simulate()))

    def test_sealed_artifacts_timing_csp_and_solid_bounds(self):
        for path,content in run.outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
        r = run.simulate()
        for row in r["latency"]["rows"]:
            self.assertLess(row["population_variance_error"],1e-10)
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


if __name__ == "__main__":
    unittest.main()
