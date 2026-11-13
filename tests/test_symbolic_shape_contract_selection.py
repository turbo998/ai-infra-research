import itertools
import json
import re
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.symbolic_shape_contract_selection import render, run


class SymbolicShapeContractTests(unittest.TestCase):
    def test_exact_solution_permutations_and_exhaustive_reference(self):
        for b,c in ((1,1),(3,7),(8,5),(12,4)):
            equations = run.fixture(b,c)
            for permuted in itertools.permutations(equations):
                self.assertEqual(run.solve(permuted),{"b":b,"c":c})
            self.assertEqual(run.exhaustive(equations,16),[{"b":b,"c":c}])
        self.assertEqual(run.simulate()["solve"]["residuals"],[0,0,0])

    def test_remainder_positive_domain_and_consistency(self):
        self.assertEqual([row["rejected"] for row in run.simulate()["negative_controls"]],[True]*3)
        scope = run.Scope((("b",3),))
        with self.assertRaises(run.ShapeError):
            run.solve([run.Equation(run.affine(scope,b=1),2)])
        self.assertEqual(run.solve([run.Equation(run.affine(scope,9,b=-2),3)]),{"b":3})
        with self.assertRaises(run.ShapeError):
            run.Equation(run.affine(scope,b=1),True)

    def test_coupled_solvable_is_not_supported_by_univariate_strategy(self):
        scope = run.Scope()
        eqs = [run.Equation(run.affine(scope,a=1,b=1),10),
               run.Equation(run.affine(scope,a=1,b=-1),2)]
        with self.assertRaises(run.Unsolvable):
            run.solve(eqs)
        self.assertEqual(run.exhaustive(eqs,16),[{"a":6,"b":4}])
        with self.assertRaises(run.ShapeError):
            run.solve([run.Equation(run.affine(scope,a=1),6),
                       run.Equation(run.affine(scope,a=1),7)])

    def test_scope_identity_and_canonical_affine_arithmetic(self):
        one,two = run.Scope(),run.Scope()
        x,y = run.affine(one,b=2),run.affine(one,b=-2)
        self.assertEqual(x.add(y).coefficients,())
        with self.assertRaises(run.ShapeError):
            x.add(run.affine(two,b=1))
        with self.assertRaises(run.ShapeError):
            run.solve([run.Equation(x,4),run.Equation(run.affine(two,b=1),2)])
        with self.assertRaises(run.ShapeError):
            run.Scope([("b",2)])

    def test_universal_proof_is_not_sample_evaluation(self):
        plain = run.affine(run.Scope(),-5,b=2)
        self.assertLess(plain.evaluate({"b":1}),0)
        self.assertGreater(plain.evaluate({"b":8}),0)
        with self.assertRaises(run.Inconclusive):
            run.prove_nonnegative(plain)
        self.assertTrue(run.prove_nonnegative(run.affine(run.Scope((("b",3),)),-5,b=2)))
        self.assertFalse(run.prove_nonnegative(run.affine(run.Scope(),b=-1)))
        self.assertTrue(run.prove_nonnegative(run.affine(run.Scope(),0)))

    def test_signature_counts_and_padding_are_distinct_from_compile_events(self):
        b = run.estimate_budget(run.DEFAULT)
        self.assertEqual((b["concrete_request_signatures"],b["bucketed_request_signatures"]),(7,3))
        self.assertEqual((b["actual_batch_rows"],b["padded_batch_rows"]),(53,80))
        self.assertEqual((b["logical_input_bytes"],b["padding_bytes"]),(6784,3456))
        self.assertIsNone(b["observed_jax_compile_events"])
        repeated = run.estimate_budget({**run.DEFAULT,"request_batches":[3,3],"bucket_multiple":1})
        self.assertEqual(repeated["concrete_request_signatures"],1)
        self.assertEqual(repeated["padding_bytes"],0)
        for patch in ({"request_batches":[]},{"request_batches":[0]},{"request_batches":[True]},
                      {"width":1.5},{"value_bytes":True},{"bucket_multiple":0},{"width":2**53}):
            with self.assertRaises(run.ShapeError):
                run.estimate_budget({**run.DEFAULT,**patch})

    def test_javascript_exact_rounding_parity_and_invalid_inputs(self):
        profiles = [run.DEFAULT,{**run.DEFAULT,"bucket_multiple":1},
                    {**run.DEFAULT,"request_batches":[3,3],"value_bytes":4},
                    {**run.DEFAULT,"request_batches":[1,9,17],"bucket_multiple":7},
                    {**run.DEFAULT,"request_batches":[2**40+1],"width":1,"value_bytes":1,"bucket_multiple":16}]
        code = render.CALCULATOR.split("const form=")[0]
        script = code+"\nconst profiles="+json.dumps(profiles)+";\nconsole.log(JSON.stringify(profiles.map(estimateBudget)));"
        script += "for(const bad of [{request_batches:[]},{request_batches:[0]},{request_batches:[true]},"
        script += "{width:1.5},{value_bytes:true},{bucket_multiple:0},{width:2**53}])"
        script += "{let ok=false;try{estimateBudget({...profiles[0],...bad});}catch(e){ok=true;}if(!ok)throw Error('accepted invalid');}"
        result = subprocess.run(["node","-e",script],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout),[run.estimate_budget(p) for p in profiles])
        self.assertIn('role="alert"',render.workbench(run.simulate()))

    def test_sealed_artifacts_timing_csp_and_solid_bounds(self):
        for path,content in run.outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
        r = run.simulate()
        for row in r["latency"]["rows"]:
            self.assertTrue(row["oracle_matches_solution"])
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
