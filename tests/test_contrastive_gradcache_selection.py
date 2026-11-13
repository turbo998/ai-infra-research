from dataclasses import replace
import json
import math
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.contrastive_gradcache_selection.run import (
    Budget, encode, estimate, gradient_cases, infonce, masks, outputs, THETA,
)


class ContrastiveGradCacheTests(unittest.TestCase):
    def test_exact_chain_rule_matches_independent_finite_difference(self):
        for seed in (3, 7, 11):
            report = gradient_cases(seed)
            by_id = {row["id"]: row for row in report["cases"]}
            for key in ("global-differentiable", "gradcache-exact-replay"):
                self.assertLess(by_id[key]["absolute_gradient_error"], report["tolerance"])
            for key in ("detached-remote-candidates", "local-negative-accumulation", "gradcache-changed-dropout"):
                self.assertGreater(by_id[key]["absolute_gradient_error"], report["tolerance"] * 100)

    def test_detached_remote_has_same_loss_but_wrong_gradient(self):
        rows = {r["id"]: r for r in gradient_cases()["cases"]}
        self.assertEqual(rows["global-differentiable"]["loss"], rows["detached-remote-candidates"]["loss"])
        self.assertNotEqual(rows["global-differentiable"]["gradient"], rows["detached-remote-candidates"]["gradient"])

    def test_embedding_derivatives_match_each_coordinate_finite_difference(self):
        embeddings, _ = encode(THETA, masks(3))
        _, analytic = infonce(embeddings)
        epsilon = 1e-5
        for i in range(8):
            for j in range(2):
                plus, minus = [row[:] for row in embeddings], [row[:] for row in embeddings]
                plus[i][j] += epsilon
                minus[i][j] -= epsilon
                numerical = (infonce(plus)[0] - infonce(minus)[0]) / (2 * epsilon)
                self.assertAlmostEqual(analytic[i][j], numerical, places=8)

    def test_memory_and_compute_tradeoff_is_explicit(self):
        result = estimate(Budget())
        self.assertEqual(result["global_pairs"], 1024)
        self.assertEqual(result["global_negatives_per_query"], 1023)
        self.assertEqual(result["local_negatives_per_query"], 127)
        self.assertLess(result["cached_modeled_bytes"], result["dense_modeled_bytes"])
        self.assertGreater(result["cached_compute_ms"], result["monolithic_compute_ms"])
        self.assertIsNone(result["network_latency_ms"])
        self.assertIsNone(result["quality"])
        self.assertEqual(result["cache_and_embedding_grad_bytes"], 2 * 1024 * 768 * 6)

    def test_invalid_budget_is_not_silently_zeroed(self):
        for changes in ({"workers": 0}, {"mini_batch": 129}, {"score_rows": 129},
                        {"dimensions": 0.5}, {"activation_mib_per_pair": math.inf}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                estimate(replace(Budget(), **changes))

    def test_generated_artifacts_match_sealed_data(self):
        for path, content in outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"), content)
            if path.name.endswith("-cards.html"):
                self.assertEqual(content.count('<section class="card"'), 7)
                self.assertNotIn("<script", content)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'", content)
            if path.suffix == ".svg":
                root = ET.fromstring(content)
                ys = [
                    float(point.split(",")[1])
                    for polygon in root.findall(".//{http://www.w3.org/2000/svg}polygon")
                    for point in polygon.attrib["points"].split()
                ]
                self.assertGreaterEqual(min(ys), 200)
                self.assertLessEqual(max(ys), 620)

    @unittest.skipUnless(shutil.which("node"), "Node.js is optional for browser-formula parity")
    def test_h5_budget_matches_python(self):
        from experiments.contrastive_gradcache_selection.render import CALCULATOR
        source = CALCULATOR.split("const form=")[0]
        budget = estimate(Budget())
        script = source + "\nconsole.log(JSON.stringify(estimateBudget(" + json.dumps(budget["input"]) + ")));"
        result = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
        for key, value in result.items():
            self.assertAlmostEqual(value, budget[key], places=6)


if __name__ == "__main__":
    unittest.main()
