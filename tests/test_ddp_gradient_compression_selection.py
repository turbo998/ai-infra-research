import json
import math
import re
import shutil
import subprocess
from dataclasses import replace
import unittest
import xml.etree.ElementTree as ET

from experiments.ddp_gradient_compression_selection.run import (
    Profile, estimate, outputs, rank_one_step, simulate, toy_feedback,
)
from experiments.ddp_gradient_compression_selection.render import CALCULATOR, DETECTION


class GradientCompressionTests(unittest.TestCase):
    def test_ring_accounting_and_strict_shape_gate(self):
        p = Profile(rows=8, columns=8, matrices=1, vector_elements=0, rank=2)
        result = estimate(p)
        self.assertFalse(result["compression_eligible"])  # equality is not a saving
        self.assertEqual(result["dense"], result["powersgd"])
        self.assertEqual(result["dense"]["ring_send_bytes_per_rank"], 4 * 64 * 126 / 64)
        self.assertEqual(estimate(replace(p, rank=1))["powersgd"]["collectives"], 2)
        self.assertEqual(estimate(Profile())["powersgd"]["collectives"], 3)

    def test_overlap_can_erase_a_large_payload_saving(self):
        slow = estimate(Profile())
        fast = estimate(Profile(bandwidth_gbps=200))
        self.assertGreater(slow["payload_compression_ratio"], 100)
        self.assertGreater(slow["powersgd_saved_ms"], 0)
        self.assertEqual(fast["powersgd_saved_ms"], 0)
        self.assertEqual(fast["timing_only_winner"], "dense")
        self.assertIsNone(slow["training_quality"])

    def test_vectors_remain_dense_and_can_reverse_the_decision(self):
        result = estimate(Profile(vector_elements=536870912))
        self.assertGreater(result["powersgd"]["payload_bytes"], 4 * 536870912)
        self.assertLess(result["bf16"]["step_ms"], result["powersgd"]["step_ms"])

    def test_rank_one_distributed_average_and_feedback_identity(self):
        gradients = [[[2.0, 4.0], [3.0, 6.0]], [[4.0, 8.0], [5.0, 10.0]]]
        zero = [[[0.0, 0.0], [0.0, 0.0]] for _ in gradients]
        recovered, residuals, _ = rank_one_step(gradients, zero, [1.0, 1.0], True)
        for i, row in enumerate([[3.0, 6.0], [4.0, 8.0]]):
            for j, value in enumerate(row):
                self.assertAlmostEqual(recovered[i][j], value)
                self.assertAlmostEqual(sum(r[i][j] for r in residuals), 0)
        for seed in (7, 11):
            with_feedback = toy_feedback(seed=seed)
            without_feedback = toy_feedback(seed=seed, error_feedback=False)
            self.assertLess(with_feedback["telescoping_residual_gap"], 1e-10)
            self.assertLess(
                with_feedback["cumulative_relative_error_pct"],
                without_feedback["cumulative_relative_error_pct"],
            )

    def test_invalid_profiles_fail_explicitly_and_single_rank_has_no_wire(self):
        for changes in (
            {"workers": 0}, {"rank": 4097}, {"rows": 1.5},
            {"vector_elements": -1}, {"bandwidth_gbps": math.nan},
            {"alpha_us": -1}, {"overlap_ms": 101}, {"backward_ms": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                estimate(replace(Profile(), **changes))
        result = estimate(Profile(workers=1))
        self.assertEqual(result["dense"]["communication_ms"], 0)

    def test_sealed_report_and_visuals_are_reproducible(self):
        for path, content in outputs().items():
            with self.subTest(path=path.name):
                self.assertEqual(path.read_text(encoding="utf-8"), content)
        report = simulate()
        self.assertEqual(report["summary"]["timing_policy_matches"], 6)
        self.assertEqual(report["summary"]["production_approved"], 0)
        self.assertFalse(report["physical_gpu_measurement"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is optional for browser-formula parity")
    def test_javascript_calculator_matches_all_sealed_profiles(self):
        cases = simulate()["scenarios"]
        source = CALCULATOR.split("const form=")[0]
        script = source + "\nconst cases=" + json.dumps([row["input"] for row in cases]) + ";"
        script += "\nconsole.log(JSON.stringify(cases.map(estimateProfile)));"
        result = subprocess.run(
            ["node", "-e", script], check=True, capture_output=True, text=True,
        )
        for expected, actual in zip(cases, json.loads(result.stdout)):
            self.assertEqual(expected["compression_eligible"], actual["compression_eligible"])
            for path in ("dense", "bf16", "powersgd"):
                for metric, value in actual[path].items():
                    self.assertAlmostEqual(expected[path][metric], value, places=6)
        invalid = source + """
let rejected=0;
for(const p of [{...BASE,rank:5000},{...BASE,bandwidth_gbps:NaN},
  {...BASE,overlap_ms:101},{...BASE,rows:1.5},{...BASE,backward_ms:0}]) {
  try {estimateProfile(p);} catch(e) {rejected++;}
}
console.log(rejected);
"""
        invalid = "const BASE=" + json.dumps(cases[0]["input"]) + ";\n" + invalid
        self.assertEqual(
            subprocess.run(["node", "-e", invalid], check=True, capture_output=True, text=True).stdout.strip(),
            "5",
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is optional for theme testing")
    def test_explicit_light_theme_is_not_overridden_by_dark_os(self):
        script = """
let theme;
const window={location:{search:"?scoutTheme=light"},matchMedia:()=>({matches:true})};
const document={documentElement:{setAttribute:(key,value)=>{theme=value;}}};
""" + DETECTION + '\nconsole.log(theme);'
        self.assertEqual(
            subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True).stdout.strip(),
            "light",
        )

    def test_report_csp_and_all_bar_coordinates_derive_from_sealed_data(self):
        generated = outputs()
        deck = next(text for path, text in generated.items() if path.name.endswith("-cards.html"))
        self.assertEqual(deck.count('<section class="card"'), 7)
        self.assertIn("width:900px;height:1125px", deck)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", deck)
        self.assertNotIn("<script", deck)
        self.assertNotRegex(deck, r'\b(?:src|href)=["\']https?://')
        widths = [float(value) for value in re.findall(r'style="width:([0-9.]+)%"', deck)]
        self.assertEqual(len(widths), 11)
        self.assertTrue(all(0 <= width <= 100 for width in widths))
        svg = next(text for path, text in generated.items() if path.suffix == ".svg")
        root = ET.fromstring(svg)
        self.assertEqual(root.attrib["viewBox"], "0 0 1600 1000")
        self.assertIn("not GPU measurements", svg)
        ys = [
            float(point.split(",")[1])
            for polygon in root.findall(".//{http://www.w3.org/2000/svg}polygon")
            for point in polygon.attrib["points"].split()
        ]
        self.assertGreaterEqual(min(ys), 200)
        self.assertLessEqual(max(ys), 620)


if __name__ == "__main__":
    unittest.main()
