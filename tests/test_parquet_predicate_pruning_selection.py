import itertools
import json
import re
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.parquet_predicate_pruning_selection import render, run


class ParquetPruningTests(unittest.TestCase):
    def test_not_may_match_loses_rows_but_truth_sets_preserve_them(self):
        r = run.simulate()["case"]
        self.assertEqual(r["selected_groups"],[0,3])
        self.assertEqual(r["exact_matching_rows"],6)
        self.assertEqual(r["pruned_matching_rows"],6)
        self.assertEqual(r["candidate_rows"],8)
        self.assertEqual(r["wrong_boolean_not_selected_groups"],[0,2])
        self.assertEqual(r["wrong_boolean_not_lost_matches"],3)

    def test_exhaustive_truth_set_soundness_for_short_nullable_columns(self):
        eq,gt,null = run.Predicate("eq","x",0),run.Predicate("gt","x",0),run.Predicate("is_null","x")
        leaves = (eq,gt,null)
        predicates = list(leaves)+[run.Predicate("not",children=(p,)) for p in leaves]
        predicates += [run.Predicate(op,children=pair) for op in ("and","or")
                       for pair in itertools.product(leaves,repeat=2)]
        for length in range(4):
            for values in itertools.product((None,-1,0,1),repeat=length):
                stats = {"x":run.summarize(values)}
                for p in predicates:
                    actual = {run.evaluate(p,{"x":v}) for v in values}
                    abstract = run.possible_truths(p,stats)
                    self.assertTrue(actual<=abstract,(values,p,actual,abstract))
                    degraded = {"x":run.ColumnStats(length,None,None,None if length else 0)}
                    self.assertTrue(actual<=run.possible_truths(p,degraded))

    def test_unknown_null_count_is_not_inferred_from_constant_bounds(self):
        null = run.Predicate("is_null","x")
        stats = {"x":run.ColumnStats(2,7,7,None)}
        self.assertEqual(run.possible_truths(null,stats),{True,False})
        self.assertEqual(run.evaluate(null,{"x":None}),True)
        self.assertEqual(run.possible_truths(null,{"x":run.ColumnStats(2,7,7,0)}),{False})
        self.assertEqual(run.sql_not(None),None)
        self.assertEqual(run.sql_and(False,None),False)
        self.assertEqual(run.sql_or(True,None),True)

    def test_missing_stats_reversal_and_correlation_only_admit_false_positives(self):
        r = run.simulate()
        self.assertEqual(r["variants"][1]["change_matches"],0)
        self.assertEqual(r["variants"][2]["selected_groups"],[0,1,2,3])
        self.assertEqual(r["variants"][2]["change_matches"],0)
        self.assertTrue(r["correlation"]["metadata_may_match"])
        self.assertEqual(r["correlation"]["actual_matching_rows"],0)
        for group in run.fixture():
            p = run.query()
            self.assertEqual(run.possible_truths(p,run.metadata_for(group)),
                             run.possible_truths(p,run.metadata_for(list(reversed(group)))))

    def test_invalid_metadata_predicates_and_missing_entries_fail_explicitly(self):
        for args in ((-1,None,None,0),(2,8,7,0),(2,7,None,0),(2,7,7,2),(2,None,None,3),
                     (True,None,None,0),(2,True,True,0)):
            with self.assertRaises(ValueError):
                run.ColumnStats(*args)
        for make in (lambda:run.Predicate("eq","x",None),lambda:run.Predicate("gt","x",True),
                     lambda:run.Predicate("not"),lambda:run.Predicate("unsupported","x"),
                     lambda:run.possible_truths(run.query(),{}),
                     lambda:run.metadata_for([{"x":1},{"y":2}]),
                     lambda:run.summarize([float("nan")]),
                     lambda:run.count_pruned(run.fixture(),[],run.query())):
            with self.assertRaises(ValueError):
                make()
        self.assertEqual(run.possible_truths(run.query(),{"x":run.summarize([])}),set())

    def test_byte_budget_keeps_metadata_for_zero_candidate_groups(self):
        b = run.estimate_budget(run.DEFAULT)
        self.assertEqual(b["selected_column_chunk_bytes"],67108864)
        self.assertEqual(b["pruned_plus_metadata_bytes"],68157440)
        self.assertEqual(b["projected_plus_metadata_bytes"],1074790400)
        zero = run.estimate_budget({**run.DEFAULT,"selected_groups":0})
        self.assertEqual(zero["selected_column_chunk_bytes"],0)
        self.assertEqual(zero["pruned_plus_metadata_bytes"],1048576)
        self.assertIsNone(zero["actual_reader_bytes"])
        for patch in ({"row_groups":0},{"selected_groups":1025},{"projected_columns":65},
                      {"columns":True},{"metadata_bytes":-1},{"column_chunk_bytes":2**53-1}):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,**patch})

    def test_javascript_exact_byte_parity_and_invalid_input_alert(self):
        profiles = [run.DEFAULT,{**run.DEFAULT,"selected_groups":0},
                    {**run.DEFAULT,"selected_groups":1024,"projected_columns":64},
                    {"row_groups":1,"selected_groups":1,"columns":1,"projected_columns":1,
                     "column_chunk_bytes":13,"metadata_bytes":0}]
        code = render.CALCULATOR.split("const form=")[0]
        script = code+"\nconst profiles="+json.dumps(profiles)+";\nconsole.log(JSON.stringify(profiles.map(estimateBudget)));"
        script += "for(const bad of [{row_groups:0},{selected_groups:1025},{projected_columns:65},"
        script += "{columns:true},{metadata_bytes:-1},{column_chunk_bytes:2**53-1}])"
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
            self.assertEqual(len(row["primary_samples_ms"]),7)
            self.assertEqual(row["primary_median_ms"],statistics.median(row["primary_samples_ms"]))
            self.assertEqual(row["baseline_median_ms"],statistics.median(row["baseline_samples_ms"]))
            self.assertEqual(row["matching_rows"],row["selected_groups"]*192)
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
