from dataclasses import replace
import itertools
import json
import math
import shutil
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.gaussian_tile_rasterization_selection.run import (
    ALPHA_FLOOR, BACKGROUND, Budget, Gaussian, alpha_gradients, closed_composite,
    composite, estimate, fixture, load_latency, maximum_error, outputs, render,
    simulate, weight,
)


class GaussianTileRasterizationTests(unittest.TestCase):
    def test_tile_candidates_match_same_support_dense_for_all_partitions_and_edges(self):
        scene = fixture(12,9,7)
        edge = Gaussian(8.5,6.5,0.5,0.75,(1,0,1),0.7,(1,0,0))
        outside = replace(edge,x=-3)
        for gaussians in (scene,scene+[edge,outside],[]):
            reference = render(gaussians,9,7,use_tiles=False)["pixels"]
            for tile in (1,2,4,8,16):
                result = render(gaussians,9,7,tile=tile)
                self.assertEqual(result["pixels"],reference)
        self.assertEqual(render([],1,1)["pixels"],[list(BACKGROUND)])

    def test_order_transmittance_background_and_independent_product_form(self):
        layers = [(0.6,(1,0,0)),(0.5,(0,0,1))]
        correct = composite(layers)
        self.assertEqual(correct["rgb"],closed_composite(layers,BACKGROUND))
        self.assertAlmostEqual(correct["render_alpha"],0.8)
        reversed_layers = composite(list(reversed(layers)))
        self.assertEqual(correct["render_alpha"],reversed_layers["render_alpha"])
        self.assertAlmostEqual(maximum_error([correct["rgb"]],[reversed_layers["rgb"]]),0.3)
        self.assertAlmostEqual(maximum_error([correct["rgb"]],[composite(layers,additive=True)["rgb"]]),0.3)
        self.assertAlmostEqual(maximum_error([correct["rgb"]],[composite(layers,omit_background=True)["rgb"]]),0.06)

    def test_effective_alpha_derivatives_match_independent_finite_differences(self):
        layers = [(0.4,(0.2,0.6,0.8)),(0.7,(0.9,0.1,0.3)),(0.2,(0.1,0.4,0.2))]
        upstream = (0.7,-0.2,0.3)
        analytic = alpha_gradients(layers,BACKGROUND,upstream)
        value = lambda data:math.fsum(a*b for a,b in zip(composite(data)["rgb"],upstream))
        for i,(alpha,color) in enumerate(layers):
            plus,minus = layers[:],layers[:]
            plus[i],minus[i] = (alpha+1e-6,color),(alpha-1e-6,color)
            self.assertAlmostEqual((value(plus)-value(minus))/2e-6,analytic[i],places=8)

    def test_exclusive_cutoff_uses_pre_step_T_and_small_alphas_can_accumulate(self):
        report = simulate()
        cutoff = report["exclusive_cutoff"]
        error = maximum_error([cutoff["exact"]["rgb"]],[cutoff["source_style_exclusive"]["rgb"]])
        self.assertGreater(error,1e-4)
        self.assertLessEqual(error,cutoff["unit_rgb_error_bound_pre_T"])
        self.assertEqual(cutoff["source_style_exclusive"]["used_layers"],2)
        self.assertAlmostEqual(cutoff["source_style_exclusive"]["transmittance"],0.0004)
        for alphas in itertools.product((0.2,0.7,0.98),repeat=3):
            layers = list(zip(alphas,[(1,0,0),(0,1,0),(0,0,1)]))
            exact,approx = composite(layers),composite(layers,cutoff=0.05)
            if approx["pre_exclusive_cutoff_transmittance"] is not None:
                self.assertLessEqual(maximum_error([exact["rgb"]],[approx["rgb"]]),
                                     approx["pre_exclusive_cutoff_transmittance"]+1e-15)
        accumulation = report["alpha_floor_accumulation"]
        self.assertGreater(accumulation["full"]["rgb"][1],0.77)
        self.assertEqual(accumulation["floor_policy"]["rgb"],[0,0,0])

    def test_opacity_cap_floor_gates_and_invalid_geometry(self):
        g = Gaussian(0.5,0.5,1,1,(1,0,1),1,(1,0,0))
        self.assertEqual(weight(g,0.5,0.5),0.99)
        self.assertEqual(weight(replace(g,opacity=ALPHA_FLOOR),0.5,0.5),ALPHA_FLOOR)
        self.assertIsNone(weight(replace(g,opacity=math.nextafter(ALPHA_FLOOR,0)),0.5,0.5))
        self.assertIsNone(weight(g,3,3))
        for invalid in (replace(g,conic=(1,2,1)),replace(g,depth=-1),
                        replace(g,color=(math.nan,0,0)),replace(g,opacity=1.1)):
            with self.assertRaises(ValueError):
                render([invalid],1,1)
        with self.assertRaises(ValueError):
            composite([(0.5,(1,0,0))],cutoff=-1)

    def test_wide_offsets_remaining_identity_widths_and_exact_payload_budget(self):
        b = estimate(Budget())
        self.assertEqual(b["intersection_entries"],8000000)
        self.assertEqual(b["two_sort_buffer_payload_bytes"],192000000)
        self.assertEqual(b["tile_offset_int64_bytes"],65280)
        self.assertEqual(b["tile_offset_int32_comparison_bytes"],32640)
        wide = estimate(replace(Budget(),gaussians_per_image=2**28))
        self.assertTrue(wide["offsets_exceed_signed_int32"])
        self.assertTrue(wide["flattened_identity_fits_signed_int32"])
        self.assertFalse(estimate(replace(Budget(),images=3,gaussians_per_image=1000000000))["flattened_identity_fits_signed_int32"])
        offsets = simulate()["wide_offsets"]
        self.assertLess(offsets["signed_int32_wrapped"][-1],0)
        self.assertFalse(offsets["large_arrays_allocated"])
        for invalid in (replace(Budget(),tile=0),replace(Budget(),intersections_per_gaussian=9000),
                        replace(Budget(),images=True),replace(Budget(),gaussians_per_image=9007199254740991)):
            with self.assertRaises(ValueError):
                estimate(invalid)

    def test_sealed_timing_medians_artifacts_csp_and_solid_geometry(self):
        for row in load_latency()["rows"]:
            for key in ("tile_indexed","dense_scan"):
                self.assertEqual(row[key+"_median_ms"],statistics.median(row["samples_ns"][key])/1e6)
        for path,content in outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
            if path.name.endswith("-cards.html"):
                self.assertEqual(content.count('<section class="card"'),7)
                self.assertNotIn("<script",content)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'",content)
            if path.suffix == ".svg":
                root = ET.fromstring(content)
                self.assertEqual(root.attrib["viewBox"],"0 0 1600 1000")
                for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
                    self.assertTrue(all(200 <= float(point.split(",")[1]) <= 620
                                        for point in polygon.attrib["points"].split()))

    @unittest.skipUnless(shutil.which("node"),"Node.js optional")
    def test_javascript_tile_ceil_wide_offset_and_metadata_budget_parity(self):
        from experiments.gaussian_tile_rasterization_selection.render import CALCULATOR
        profiles = [Budget(),replace(Budget(),gaussians_per_image=2**28),
                    replace(Budget(),width=17,height=9,tile=8,intersections_per_gaussian=2),
                    replace(Budget(),gaussians_per_image=0,intersections_per_gaussian=0),
                    replace(Budget(),width=2**32+1,height=1,tile=1,
                            gaussians_per_image=0,intersections_per_gaussian=0)]
        source = CALCULATOR.split("const form=")[0]
        script = source+"\nconsole.log(JSON.stringify("+json.dumps([estimate(p)["input"] for p in profiles])+".map(estimateBudget)));"
        actual = json.loads(subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)
        for profile,result in zip(profiles,actual):
            for key,value in result.items():
                self.assertEqual(value,estimate(profile)[key])
        invalid = {**estimate(Budget())["input"],"intersections_per_gaussian":9000}
        script = source+"\ntry{estimateBudget("+json.dumps(invalid)+");process.exit(1)}catch(e){console.log(e.message)}"
        self.assertIn("tile",subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)


if __name__ == "__main__":
    unittest.main()
