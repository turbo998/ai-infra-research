import json
import math
import re
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.rope_layout_partial_dimension_selection import render, run


class RoPELayoutTests(unittest.TestCase):
    def test_partial_rotation_matches_independent_complex_relative_oracle(self):
        for width,rotary in ((4,2),(8,6),(8,8),(16,10)):
            q,k = [math.sin(i+.2) for i in range(width)],[math.cos(i-.4) for i in range(width)]
            for m,n in ((0,0),(0,7),(3,9),(129,7),(-3,4)):
                q_rot,k_rot = run.rotate(q,m,rotary),run.rotate(k,n,rotary)
                self.assertAlmostEqual(run.dot(q_rot,k_rot),
                                       run.complex_relative_oracle(q,k,m,n,rotary),places=11)
                self.assertEqual(q_rot[rotary:],q[rotary:])
                self.assertEqual(k_rot[rotary:],k[rotary:])

    def test_layout_conversion_commutes_with_rotation_and_preserves_cross_score(self):
        r = run.simulate()["case"]
        q,k,d = r["query"],r["key"],r["rotary_dim"]
        self.assertEqual(run.to_interleaved(run.to_split(q,d),d),q)
        converted = run.rotate(run.to_split(q,d),r["query_position"],d,False)
        expected = run.to_split(run.rotate(q,r["query_position"],d),d)
        for a,b in zip(converted,expected):
            self.assertAlmostEqual(a,b,places=13)
        self.assertLess(r["converted_absolute_error"],r["tolerance"])
        self.assertGreater(r["wrong_layout_absolute_error"],r["tolerance"])

    def test_same_position_dot_checks_do_not_detect_the_wrong_layout(self):
        r = run.simulate()["case"]
        for control in r["same_position_controls"]:
            self.assertAlmostEqual(control["score"],control["unrotated_score"],places=12)
        self.assertGreater(r["wrong_layout_absolute_error"],r["tolerance"])

    def test_shared_position_shift_invariance_and_double_rotation_control(self):
        r = run.simulate()
        for row in r["variants"]:
            self.assertLess(row["correct_absolute_error"],r["case"]["tolerance"])
            self.assertGreater(row["wrong_layout_absolute_error"],r["case"]["tolerance"])
        self.assertGreater(r["case"]["double_rotation_absolute_error"],r["case"]["tolerance"])
        q = r["case"]["query"]
        rotated = run.rotate(q,3,6)
        self.assertAlmostEqual(run.dot(q,q),run.dot(rotated,rotated),places=12)

    def test_dimension_coefficient_and_layout_errors_are_explicit(self):
        for operation in (lambda:run.rotate([1,2],0,4),lambda:run.rotate([1,2,3],0,3),
                          lambda:run.rotate([True,2],0,2),lambda:run.rotate([1,float("nan")],0,2),
                          lambda:run.rotate([1,2],.5,2),lambda:run.rotate([1,2],0,2,"interleaved"),
                          lambda:run.apply_coefficients([1,2],2,(),()),
                          lambda:run.apply_coefficients([1,2],2,(1,),(float("inf"),)),
                          lambda:run.frequencies(2,1),lambda:run.dot([1],[1,2])):
            with self.assertRaises(ValueError):
                operation()

    def test_cache_and_full_qk_payload_are_separate_from_extra_allocation(self):
        b = run.estimate_budget(run.DEFAULT)
        self.assertEqual(b["selected_cos_sin_cache_bytes"],33554432)
        self.assertEqual(b["selected_full_qk_payload_bytes"],41943040)
        self.assertEqual(b["selected_rotated_prefix_payload_bytes"],20971520)
        self.assertEqual(b["selected_untouched_tail_payload_bytes"],20971520)
        self.assertIsNone(b["extra_allocation_bytes"])
        zero = run.estimate_budget({**run.DEFAULT,"active_rows":0})
        self.assertEqual(zero["selected_full_qk_payload_bytes"],0)
        self.assertEqual(zero["selected_cos_sin_cache_bytes"],b["selected_cos_sin_cache_bytes"])
        for patch in ({"rotary_dim":129},{"rotary_dim":3},{"active_rows":-1},{"cache_instances":True},
                      {"cache_element_bytes":1},{"cache_element_bytes":2},{"max_positions":2**53-1}):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,**patch})

    def test_javascript_exact_byte_parity_and_invalid_input_alert(self):
        profiles = [run.DEFAULT,{**run.DEFAULT,"active_rows":0},
                    {**run.DEFAULT,"rotary_dim":128,"cache_instances":3},
                    {**run.DEFAULT,"head_dim":8,"rotary_dim":6,"active_rows":1,"query_heads":1,"key_heads":1}]
        code = render.CALCULATOR.split("const form=")[0]
        script = code+"\nconst profiles="+json.dumps(profiles)+";\nconsole.log(JSON.stringify(profiles.map(estimateBudget)));"
        script += "for(const bad of [{rotary_dim:129},{rotary_dim:3},{active_rows:-1},{cache_instances:true},"
        script += "{cache_element_bytes:1},{cache_element_bytes:2},{max_positions:2**53-1}])"
        script += "{let ok=false;try{estimateBudget({...profiles[0],...bad});}catch(e){ok=true;}if(!ok)throw Error('accepted invalid');}"
        result = subprocess.run(["node","-e",script],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout),[run.estimate_budget(p) for p in profiles])
        self.assertIn('role="alert"',render.workbench(run.simulate()))

    def test_sealed_artifacts_timing_csp_and_solid_bounds(self):
        for path,content in run.outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
        r = run.simulate()
        for row in r["latency"]["rows"]:
            self.assertLess(row["maximum_vector_error"],1e-12)
            self.assertEqual(len(row["primary_samples_ms"]),7)
            self.assertEqual(row["primary_median_ms"],statistics.median(row["primary_samples_ms"]))
            self.assertEqual(row["baseline_median_ms"],statistics.median(row["baseline_samples_ms"]))
        deck = render.deck(r)
        self.assertEqual(deck.count('<section class="card"'),7)
        self.assertEqual(deck.count("<table>"),deck.count("</table>"))
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
