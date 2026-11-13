import json
import math
import re
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.sparse_conv_coordinate_map_selection import render, run


class SparseCoordinateMapTests(unittest.TestCase):
    def test_regular_and_submanifold_topology(self):
        t = run.simulate()["topology"]
        self.assertEqual(t["regular_output"], [1,3,3,2])
        self.assertEqual(t["submanifold_output"], [3,3])
        self.assertEqual(t["second_regular_at_input"], [7,8])
        self.assertEqual(t["second_submanifold_output"], [6,6])
        self.assertEqual(t["zero_feature_output"], [0,0])
        self.assertEqual(t["zero_feature_active_count"], 2)

    def test_independent_dense_strides_dilation_and_batch_isolation(self):
        self.assertTrue(all(row["maximum_error"] == 0 for row in run.simulate()["variants"]))
        coordinates = [(0,2),(1,2)]
        mapping = run.build_map(coordinates,8,2,submanifold=True)
        self.assertEqual(run.apply_map(mapping,[1,7],[1,1,1]), [1,7])
        empty = run.build_map([],8)
        self.assertEqual(run.apply_map(empty,[],[1,2,3]), [])
        self.assertEqual(run.dense_reference(empty,[],[1,2,3]), [])

    def test_cached_adjoint_dot_identity_finite_differences_not_inverse(self):
        inv = run.simulate()["inverse"]
        self.assertEqual(inv["cached_tied_weight_adjoint"], [0.5,4])
        self.assertEqual(inv["dot_left"], inv["dot_right"])
        self.assertLess(inv["maximum_gradient_error"], 1e-8)
        self.assertEqual(inv["adjoint_of_forward"], [20,-4])
        self.assertNotEqual(inv["adjoint_of_forward"], inv["features"])

    def test_transposed_support_is_not_cached_inverse_support(self):
        inv = run.simulate()["inverse"]
        self.assertEqual(len(inv["input_coordinates"]), 2)
        self.assertEqual(len(inv["ordinary_transposed"]["coordinates"]), 6)
        full = dict(zip(map(tuple,inv["ordinary_transposed"]["coordinates"]),inv["ordinary_transposed"]["values"]))
        self.assertEqual([full[tuple(p)] for p in inv["input_coordinates"]], inv["cached_tied_weight_adjoint"])

    def test_ordered_cache_identity_and_invalid_coordinates(self):
        r = run.simulate()
        self.assertEqual(r["cache"]["fresh_output"], r["inverse"]["forward"])
        self.assertEqual(r["cache"]["maximum_stale_error"], 9)
        mapping = run.build_map([(0,1),(0,4)],8,stride=2)
        with self.assertRaisesRegex(ValueError,"ordered input"):
            run.apply_map(mapping,[-1,2],[1,2,3],[(0,4),(0,1)])
        with self.assertRaisesRegex(ValueError,"ordered output"):
            run.adjoint(mapping,[1,2,3],[1,2,3],list(reversed(mapping.outputs)))
        for coordinates in ([(0,1),(0,1)],[(1,1)],[(0,8)],[(0,-1)],[(0,True)]):
            with self.assertRaises(ValueError):
                run.build_map(coordinates,8)
        with self.assertRaises(ValueError):
            run.build_map([(0,1)],8,stride=2,submanifold=True)
        with self.assertRaises(ValueError):
            run.apply_map(mapping,[math.nan,1],[1,2,3])

    def test_exact_budget_payloads_widths_and_rejections(self):
        b = run.estimate_budget(run.DEFAULT)
        self.assertEqual(b["native_padded_pair_bytes"], 43200000)
        self.assertEqual(b["compact_valid_pair_id_bytes"], 14400000)
        self.assertEqual(b["input_coordinate_bytes"], 3200000)
        self.assertEqual(b["input_feature_bytes"], 25600000)
        wide = run.estimate_budget({**run.DEFAULT,"edge":2048})
        self.assertTrue(wide["source_selects_int64_hash_keys"])
        self.assertEqual(wide["native_padded_pair_bytes"], b["native_padded_pair_bytes"])
        for key,value in (("active_input",True),("valid_pairs",5400001),("feature_bytes",3),
                          ("active_output",2**31),("edge",2**53)):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,key:value})

    def test_sealed_artifacts_csp_svg_and_measured_samples(self):
        for path,content in run.outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
        r = run.simulate()
        for row in r["latency"]["rows"]:
            self.assertEqual(len(row["primary_samples_ms"]),7)
            self.assertEqual(row["primary_median_ms"],statistics.median(row["primary_samples_ms"]))
            self.assertEqual(row["baseline_median_ms"],statistics.median(row["baseline_samples_ms"]))
            self.assertEqual(row["maximum_error"],0)
        deck = render.deck(r)
        self.assertEqual(deck.count('<section class="card"'),7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",deck)
        self.assertNotIn("<script",deck)
        self.assertNotRegex(deck,r'(?:src|href)="https?://')
        svg = render.summary_svg(r)
        root = ET.fromstring(svg)
        self.assertEqual((root.attrib["width"],root.attrib["height"]),("1600","1000"))
        for points in re.findall(r'<polygon points="([^"]+)',svg):
            for point in points.split():
                y = float(point.split(",")[1])
                self.assertGreaterEqual(y,200)
                self.assertLessEqual(y,620)

    def test_javascript_calculator_parity_and_invalid_input(self):
        profiles = [run.DEFAULT,{**run.DEFAULT,"edge":2048},
                    {**run.DEFAULT,"index_bytes":8},
                    {**run.DEFAULT,"active_input":0,"active_output":0,"valid_pairs":0},
                    {**run.DEFAULT,"dimensions":2,"edge":1024,"feature_bytes":4}]
        code = render.CALCULATOR.split("const form=")[0]
        script = code+"\nconst profiles="+json.dumps(profiles)+";\n"
        script += "console.log(JSON.stringify(profiles.map(estimateBudget)));\n"
        script += "for(const bad of [{active_input:true},{valid_pairs:5400001},{feature_bytes:3},{edge:2**53}])"
        script += "{let rejected=false;try{estimateBudget({...profiles[0],...bad});}catch(e){rejected=true;}"
        script += "if(!rejected)throw Error('invalid accepted');}"
        result = subprocess.run(["node","-e",script],check=True,text=True,capture_output=True)
        self.assertEqual(json.loads(result.stdout),[run.estimate_budget(p) for p in profiles])
        self.assertIn('role="alert"',render.workbench(run.simulate()))


if __name__ == "__main__":
    unittest.main()
