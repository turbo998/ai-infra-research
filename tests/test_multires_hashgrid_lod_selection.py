import json
import math
import re
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.multires_hashgrid_lod_selection import render, run


class HashGridLoDTests(unittest.TestCase):
    def test_constant_partition_and_boundary_coordinates(self):
        for resolution, capacity in ((4, 16), (4, 1), (8, 8)):
            points = [(0, 0), (1, 1), (0.23, 0.41), (0, 1)]
            self.assertTrue(all(abs(v - 0.7) < 1e-14 for v in
                                run.direct_encode([0.7]*capacity, resolution, points)))
            for point in points:
                self.assertAlmostEqual(sum(t[1] for t in run.corner_terms(point, resolution, capacity)), 1)
        gradient = run.vjp([0.7], 8, [(0.23, 0.41)], [1])
        self.assertAlmostEqual(gradient["parameter"][0], 1)
        for value in gradient["coordinates"][0]:
            self.assertAlmostEqual(value, 0)

    def test_unfolded_independent_reference(self):
        for resolution, capacity in ((4, 16), (4, 4), (8, 8), (16, 32)):
            table = [math.sin(i * 0.7) for i in range(capacity)]
            points = run.fixtures(31) + [(0, 0), (1, 1), (0, 1), (1, 0)]
            for a, b in zip(run.direct_encode(table, resolution, points),
                            run.expanded_encode(table, resolution, points)):
                self.assertAlmostEqual(a, b, places=14)
        with self.assertRaises(ValueError):
            run.direct_encode([math.nan], 4, [(0.2, 0.3)])
        with self.assertRaises(ValueError):
            run.direct_encode([1], 4, [(1.01, 0.3)])

    def test_parameter_and_coordinate_finite_differences(self):
        gradient = run.simulate()["gradient"]
        self.assertLess(gradient["maximum_parameter_error"], 1e-8)
        self.assertLess(gradient["maximum_coordinate_error"], 1e-8)
        with self.assertRaises(ValueError):
            run.vjp([1], 4, [(0.2, 0.3)], [])

    def test_collision_requires_every_occurrence(self):
        c = run.simulate()["collision"]
        self.assertEqual(len(set(term[0] for term in c["corner_terms"])), 2)
        self.assertAlmostEqual(c["output"], 0.77834)
        self.assertAlmostEqual(c["parameter_gradient"][0], 0.3574)
        self.assertAlmostEqual(c["parameter_gradient"][3], 0.6426)
        self.assertAlmostEqual(c["overwrite_max_error"], 0.5913)
        self.assertNotEqual(c["parameter_gradient"], c["wrong_overwrite_gradient"])

    def test_soft_adjoint_and_consistent_epsilon_boundary(self):
        lod = run.simulate()["lod"]
        self.assertLess(lod["maximum_gradient_error"], 1e-8)
        self.assertNotAlmostEqual(lod["correct_gradient"][1], lod["wrong_missing_weight"][1])
        self.assertNotAlmostEqual(lod["correct_gradient"][1], lod["wrong_double_weight"][1])
        self.assertEqual(lod["zero_fraction_hard"], [1, 0, 0])
        self.assertEqual(lod["zero_fraction_soft"], [0.001, 0, 0])
        boundary = (1 - run.EPSILON) / 4
        self.assertEqual(run.level_weights(4, boundary, "hard"), [1, 0, 0, 0])
        self.assertEqual(run.level_weights(4, boundary - 1e-7, "hard"), [1, 0, 0, 0])
        self.assertEqual(run.level_weights(4, boundary + 1e-7, "hard"), [1, 1, 0, 0])
        for weight, raw in zip(lod["soft_weights"], lod["raw"]):
            h = 1e-6
            finite_hessian = (weight**2*(raw+h) - weight**2*(raw-h))/(2*h)
            self.assertAlmostEqual(finite_hessian, weight**2, places=8)
        for value in (-0.1, 1.01, math.nan):
            with self.assertRaises(ValueError):
                run.level_weights(3, value, "soft")

    def test_allocation_alignment_lod_independence_and_bounds(self):
        default = run.estimate_budget(run.DEFAULT)
        off = run.estimate_budget({**run.DEFAULT, "lod": 0})
        self.assertEqual(default["parameter_bytes"], 20070400)
        self.assertEqual(default["parameter_bytes"], off["parameter_bytes"])
        self.assertEqual(default["soft_unweighted_base_output_bytes"], 3145728)
        aligned = run.estimate_budget({**run.DEFAULT, "dimensions": 2, "levels": 1,
                                      "base_resolution": 3, "log2_cap": 10})
        self.assertEqual(aligned["allocated_table_entries"], 16)
        for key, value in (("levels", True), ("levels", 33), ("parameter_bytes", 3),
                           ("growth", 3), ("batch", 2**53), ("levels", 32)):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT, key: value})

    def test_sealed_artifacts_csp_geometry_and_latency(self):
        for path, content in run.outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"), content)
        r = run.simulate()
        for row in r["latency"]["rows"]:
            self.assertEqual(len(row["primary_samples_ms"]), 7)
            self.assertEqual(row["primary_median_ms"], statistics.median(row["primary_samples_ms"]))
            self.assertEqual(row["baseline_median_ms"], statistics.median(row["baseline_samples_ms"]))
            self.assertLess(row["maximum_error"], 1e-12)
        deck = render.deck(r)
        self.assertEqual(deck.count('<section class="card"'), 7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", deck)
        self.assertNotIn("<script", deck)
        self.assertNotRegex(deck, r'(?:src|href)="https?://')
        svg = render.summary_svg(r)
        root = ET.fromstring(svg)
        self.assertEqual((root.attrib["width"], root.attrib["height"]), ("1600", "1000"))
        for points in re.findall(r'<polygon points="([^"]+)', svg):
            for point in points.split():
                y = float(point.split(",")[1])
                self.assertGreaterEqual(y, 200)
                self.assertLessEqual(y, 620)

    def test_javascript_exact_calculator_and_rejections(self):
        profiles = [run.DEFAULT,
                    {**run.DEFAULT, "lod": 0},
                    {**run.DEFAULT, "dimensions": 2, "levels": 1, "base_resolution": 3},
                    {**run.DEFAULT, "levels": 4, "lod": (1-run.EPSILON)/4},
                    {**run.DEFAULT, "growth": 1, "levels": 32, "log2_cap": 24, "features": 8},
                    {**run.DEFAULT, "batch": 1, "parameter_bytes": 4}]
        code = render.CALCULATOR.split('const form=')[0]
        script = code + "\nconst profiles=" + json.dumps(profiles) + ";\n"
        script += "console.log(JSON.stringify(profiles.map(estimateBudget)));\n"
        script += "for(const bad of [{lod:NaN},{levels:true},{parameter_bytes:3},{batch:2**53},{levels:32}])"
        script += "{let rejected=false;try{estimateBudget({...profiles[0],...bad});}catch(e){rejected=true;}"
        script += "if(!rejected)throw Error('invalid accepted');}"
        result = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
        actual = json.loads(result.stdout)
        self.assertEqual(actual, [run.estimate_budget(p) for p in profiles])
        self.assertIn('role="alert"', render.workbench(run.simulate()))


if __name__ == "__main__":
    unittest.main()
