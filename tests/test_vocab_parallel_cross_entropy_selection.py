from dataclasses import replace
import json
import math
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.vocab_parallel_cross_entropy_selection.run import (
    Budget, dense_reference, estimate, LOGITS, TARGETS, partitioned_loss, outputs, simulate,
)


class VocabParallelCrossEntropyTests(unittest.TestCase):
    def test_exact_distributed_statistics_match_dense_and_finite_differences(self):
        result = partitioned_loss(LOGITS, TARGETS, 10, 3)
        reference = dense_reference(LOGITS, TARGETS, 10)
        self.assertAlmostEqual(result["loss"], reference["loss"], places=12)
        epsilon = 1e-3
        for i in range(len(LOGITS)):
            for j in range(len(LOGITS[0])):
                plus, minus = [row[:] for row in LOGITS], [row[:] for row in LOGITS]
                plus[i][j] += epsilon
                minus[i][j] -= epsilon
                numerical = (dense_reference(plus, TARGETS, 10)["loss"] - dense_reference(minus, TARGETS, 10)["loss"]) / (2 * epsilon)
                self.assertAlmostEqual(result["gradients"][i][j], numerical, places=7)

    def test_wrong_controls_are_detected_even_when_forward_loss_is_equal(self):
        data = simulate()
        rows = {r["id"]: r for r in data["cases"]}
        self.assertTrue(rows["exact"]["matches_reference"])
        for name in ("average-denominator", "target-on-every-shard", "unmasked-padding", "owner-local-softmax"):
            self.assertFalse(rows[name]["matches_reference"])
            self.assertGreater(rows[name]["maximum_gradient_error"], 0.01)
        self.assertEqual(rows["target-on-every-shard"]["absolute_loss_error"], 0)
        self.assertTrue(all(r["matches_reference"] for r in data["perturbations"]))

    def test_ignored_tokens_padded_classes_and_empty_denominator_policy(self):
        result = partitioned_loss(LOGITS, TARGETS, 10, 3)
        self.assertEqual(result["gradients"][-1], [0.0] * 12)
        self.assertTrue(all(row[10:] == [0.0, 0.0] for row in result["gradients"]))
        for row in result["gradients"]:
            self.assertAlmostEqual(sum(row), 0)
        empty = partitioned_loss(LOGITS, [-100]*4, 10, 3)
        self.assertIsNone(empty["loss"])
        self.assertIn("skip optimizer step", empty["status"])
        with self.assertRaises(ValueError):
            partitioned_loss(LOGITS, [10, 4, 9, -100], 10, 3)

    def test_ring_payload_startup_and_single_rank_unknown_ratio(self):
        result = estimate(Budget())
        self.assertEqual(result["gathered_fp32_logits_bytes"], 2**31)
        self.assertEqual(result["local_fp32_logits_bytes"], 2**28)
        self.assertEqual(result["stats_send_bytes_per_rank"], 3 * 4 * 4096 * 14 / 8)
        tiny = estimate(Budget(tokens=1))
        self.assertGreater(tiny["stats_modeled_ms"], tiny["allgather_modeled_ms"])
        single = estimate(Budget(partitions=1))
        self.assertIsNone(single["forward_wire_ratio"])
        self.assertEqual(single["stats_modeled_ms"], 0)
        self.assertEqual(estimate(Budget(vocabulary=10, partitions=3))["padding_classes"], 2)

    def test_invalid_budget_and_loss_inputs_fail_explicitly(self):
        for changes in ({"tokens": 0}, {"partitions": 131073}, {"vocabulary": 1.5}, {"bandwidth_gbps": math.nan}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                estimate(replace(Budget(), **changes))
        with self.assertRaises(ValueError):
            partitioned_loss(LOGITS, TARGETS, 10, 5)
        with self.assertRaises(ValueError):
            partitioned_loss(LOGITS, TARGETS, 10, 3, order=[0, 0, 1])

    def test_generated_seven_card_deck_and_svg_geometry(self):
        for path, content in outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"), content)
            if path.name.endswith("-cards.html"):
                self.assertEqual(content.count('<section class="card"'), 7)
                self.assertNotIn("<script", content)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'", content)
            if path.suffix == ".svg":
                root = ET.fromstring(content)
                self.assertEqual(root.attrib["viewBox"], "0 0 1600 1000")
                for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
                    for point in polygon.attrib["points"].split():
                        y = float(point.split(",")[1])
                        self.assertGreaterEqual(y, 200)
                        self.assertLessEqual(y, 620)

    @unittest.skipUnless(shutil.which("node"), "Node.js is optional for browser parity")
    def test_browser_budget_matches_python(self):
        from experiments.vocab_parallel_cross_entropy_selection.render import CALCULATOR
        source = CALCULATOR.split("const form=")[0]
        budgets = [estimate(Budget()), estimate(Budget(tokens=1)), estimate(Budget(partitions=1))]
        script = source + "\nconsole.log(JSON.stringify(" + json.dumps([r["input"] for r in budgets]) + ".map(estimateBudget)));"
        actual = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
        for expected, result in zip(budgets, actual):
            for key, value in result.items():
                if value is None:
                    self.assertIsNone(expected[key])
                else:
                    self.assertAlmostEqual(value, expected[key], places=6)


if __name__ == "__main__":
    unittest.main()
