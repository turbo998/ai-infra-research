import itertools
import json
import re
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.distributed_auroc_threshold_selection import render, run


class DistributedAUROCTests(unittest.TestCase):
    def test_global_auc_contains_cross_rank_pairs(self):
        c = run.simulate()["case"]
        self.assertEqual(c["local_aucs"],[1,1])
        self.assertEqual(c["exact_auc"],.75)
        self.assertEqual(c["exact_auc"],c["pairwise_oracle_auc"])
        self.assertNotEqual(c["mean_local_auc"],c["exact_auc"])

    def test_tied_score_permutations_and_independent_pairwise_credit(self):
        pairs = [(0.1,0),(0.2,1),(0.2,0),(0.9,1)]
        for permuted in itertools.permutations(pairs):
            self.assertEqual(run.exact_roc(permuted)[0],.875)
            self.assertEqual(run.pairwise_auc(permuted),.875)
        for labels in itertools.product((0,1),repeat=5):
            if not any(labels) or all(labels):
                continue
            data = list(zip((0,.25,.25,.5,1),labels))
            self.assertEqual(run.exact_roc(data)[0],run.pairwise_auc(data))

    def test_threshold_grid_merge_empty_identity_and_boundary_equality(self):
        pairs = [(0.1,0),(0.2,1),(0.8,0),(0.9,1)]
        grid = (0,.5,1)
        whole = run.threshold_counts(pairs,grid)
        for split in range(5):
            states = [run.threshold_counts(pairs[:split],grid),run.threshold_counts([],grid),
                      run.threshold_counts(pairs[split:],grid)]
            self.assertEqual(run.merge_states(states),whole)
        at_boundary = run.threshold_counts([(.5,0),(.5,1)],grid)
        self.assertEqual(at_boundary.cells[1],(0,1,0,1))
        with self.assertRaises(ValueError):
            run.merge_states([whole,run.threshold_counts(pairs,(0,.6,1))])

    def test_coarse_approximation_and_source_binned_endpoint_convention(self):
        r = run.simulate()
        self.assertEqual((r["case"]["coarse_auc"],r["case"]["fine_auc"]),(.5,.75))
        e = r["endpoint"]
        self.assertEqual(e["exact_auc"],.5)
        self.assertEqual(e["inspected_binned_rule_auc"],0)
        self.assertFalse(e["origin_present"])
        self.assertEqual(e["original_explicit_origin_completion_auc"],.5)

    def test_undefined_classes_invalid_scores_and_thresholds_are_explicit(self):
        for pairs in ([],[(.1,0)],[(.9,1)]):
            with self.assertRaises(run.UndefinedMetric):
                run.exact_roc(pairs)
            with self.assertRaises(run.UndefinedMetric):
                run.threshold_roc(run.threshold_counts(pairs,(0,1)))
        for pairs in ([(-1,0)],[(float("nan"),1)],[(.5,2)],[(.5,True)]):
            with self.assertRaises(ValueError):
                run.threshold_counts(pairs,(0,1))
        for grid in ((0,.5,.5,1),(.1,1),(0,)):
            with self.assertRaises(ValueError):
                run.threshold_counts([(.5,1)],grid)

    def test_state_payload_and_update_workspace_are_separate(self):
        b = run.estimate_budget(run.DEFAULT)
        self.assertEqual(b["gathered_raw_payload_bytes_per_rank"],12000000)
        self.assertEqual(b["fixed_counter_state_bytes_per_rank"],4096)
        self.assertEqual(b["selected_vectorized_int64_matrix_bytes"],51200000)
        loop = run.estimate_budget({**run.DEFAULT,"update_samples":50001})
        self.assertEqual(loop["source_update_path"],"threshold loop")
        self.assertIsNone(loop["selected_vectorized_int64_matrix_bytes"])
        self.assertEqual(loop["fixed_counter_state_bytes_per_rank"],b["fixed_counter_state_bytes_per_rank"])
        for patch in ({"rank_counts":[]},{"rank_counts":[0,0]},{"threshold_count":1},
                      {"prediction_bytes":True},{"target_bytes":2},{"update_samples":250001},
                      {"threshold_count":2**53}):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,**patch})

    def test_javascript_exact_byte_and_branch_parity(self):
        profiles = [run.DEFAULT,{**run.DEFAULT,"update_samples":50001},
                    {**run.DEFAULT,"rank_counts":[0,1,3],"update_samples":1,"threshold_count":2},
                    {**run.DEFAULT,"prediction_bytes":8,"threshold_bytes":8},
                    {**run.DEFAULT,"update_samples":0,"threshold_count":2048}]
        code = render.CALCULATOR.split("const form=")[0]
        script = code+"\nconst profiles="+json.dumps(profiles)+";\nconsole.log(JSON.stringify(profiles.map(estimateBudget)));"
        script += "for(const bad of [{rank_counts:[]},{rank_counts:[0,0]},{threshold_count:1},"
        script += "{prediction_bytes:true},{target_bytes:2},{update_samples:250001},{threshold_count:2**53}])"
        script += "{let ok=false;try{estimateBudget({...profiles[0],...bad});}catch(e){ok=true;}if(!ok)throw Error('accepted invalid');}"
        result = subprocess.run(["node","-e",script],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout),[run.estimate_budget(p) for p in profiles])
        self.assertIn('role="alert"',render.workbench(run.simulate()))

    def test_sealed_artifacts_timing_csp_and_solid_bounds(self):
        for path,content in run.outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
        r = run.simulate()
        for row in r["latency"]["rows"]:
            self.assertEqual(row["maximum_auc_error"],0)
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
