from dataclasses import replace
import json
import math
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.ghost_clipping_selection.run import (
    ACTIVATIONS, PARAMETERS, TARGETS, Budget, batch, clipped_sum, coefficient,
    estimate, example, finite_difference, ghost_norm, norm, outputs, partition_sum,
    simulate, weighted_loss,
)


class GhostClippingTests(unittest.TestCase):
    def test_per_example_materialized_gradients_match_independent_finite_differences(self):
        for activations,targets in zip(ACTIVATIONS,TARGETS):
            reference = example(PARAMETERS,activations,targets)
            numerical = finite_difference(lambda p:example(p,activations,targets)["loss"])
            for a,b in zip(reference["gradient"],numerical):
                self.assertAlmostEqual(a,b,places=8)

    def test_sequence_gram_norms_keep_alignment_cancellation_and_bias(self):
        for activations,row in zip(ACTIVATIONS,batch()):
            self.assertAlmostEqual(ghost_norm(activations,row["backprops"]),norm(row["gradient"]),places=12)
        self.assertEqual(norm(batch()[1]["gradient"]),0)
        single = example(PARAMETERS,[ACTIVATIONS[2][0]],[TARGETS[2][0]])
        self.assertAlmostEqual(
            ghost_norm([ACTIVATIONS[2][0]],single["backprops"],"gram"),
            ghost_norm([ACTIVATIONS[2][0]],single["backprops"],"frobenius-product"),places=12)
        report = simulate()
        cases = {row["id"]:row for row in report["cases"]}
        self.assertGreater(max(cases["diagonal-only"]["individual_clipped_norms"]),2.3)
        self.assertGreater(max(cases["independent-weight-bias-bounds"]["individual_clipped_norms"]),2.8)
        self.assertLess(max(cases["sequence-gram-ghost"]["individual_clipped_norms"]),2)

    def test_stopped_weighted_replay_matches_clipped_sum_but_recomputed_factors_do_not(self):
        rows = batch()
        factors = [coefficient(norm(row["gradient"])) for row in rows]
        reference = clipped_sum([row["gradient"] for row in rows],factors)
        numerical = finite_difference(lambda p:weighted_loss(p,factors))
        for a,b in zip(reference,numerical):
            self.assertAlmostEqual(a,b,places=8)
        cases = {row["id"]:row for row in simulate()["cases"]}
        self.assertTrue(cases["stopped-coefficient-replay"]["matches_reference"])
        for name in ("differentiate-through-coefficients","clip-after-example-aggregation","frobenius-product"):
            self.assertFalse(cases[name]["matches_reference"])
            self.assertGreater(cases[name]["maximum_absolute_error"],0.2)

    def test_physical_partitioning_preserves_the_logical_clipped_sum(self):
        report = simulate()
        for row in report["partition_checks"]:
            self.assertLess(row["maximum_absolute_error"],1e-12)
        gradients = [row["gradient"] for row in batch()]
        with self.assertRaises(ValueError):
            partition_sum(gradients,[1,1])
        self.assertEqual(coefficient(0),1)
        self.assertEqual(clipped_sum([[0.0]*6],[1]),[0.0]*6)

    def test_synthetic_noise_realization_and_expected_denominator_do_not_claim_privacy(self):
        release = simulate()["synthetic_release"]
        self.assertFalse(release["noise_is_gaussian_draw"])
        self.assertFalse(release["secure_rng"])
        self.assertIsNone(release["privacy_epsilon"])
        self.assertIsNone(release["privacy_delta"])
        self.assertEqual(release["mean_noise_std_if_gaussian"],0.5)
        for right,wrong in zip(release["release"],release["wrong_actual_count_release"]):
            self.assertAlmostEqual(wrong,right*4/3,places=12)
        b = estimate(Budget())
        self.assertAlmostEqual(b["repeated_chunk_noise_std"]**2,
                               b["physical_chunks"]*b["mean_release_noise_std"]**2,places=12)
        self.assertEqual(estimate(replace(Budget(),noise_multiplier=0))["mean_release_noise_std"],0)

    def test_partial_memory_shape_crossover_and_invalid_inputs(self):
        default = estimate(Budget())
        self.assertEqual(default["per_example_weight_gradient_bytes"],64*4096*4096*4)
        self.assertEqual(default["two_sequence_gram_bytes"],2*64*128*128*4)
        self.assertEqual(default["dense_to_gram_ratio"],512)
        long = estimate(replace(Budget(),tokens=4096))
        self.assertGreater(long["two_sequence_gram_bytes"],long["per_example_weight_gradient_bytes"])
        uneven = estimate(replace(Budget(),logical_examples=513))
        self.assertEqual(uneven["physical_chunks"],9)
        for profile in (replace(Budget(),physical_examples=513),replace(Budget(),clip=0),
                        replace(Budget(),noise_multiplier=-1),replace(Budget(),assumed_read_gbps=math.inf),
                        replace(Budget(),noise_multiplier=1e308,clip=1e308)):
            with self.assertRaises(ValueError):
                estimate(profile)
        with self.assertRaises(ValueError):
            ghost_norm([],[])
        with self.assertRaises(ValueError):
            example(PARAMETERS,ACTIVATIONS[0],[])

    def test_reproducible_artifacts_csp_seven_cards_and_svg_bounds(self):
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
    def test_javascript_memory_noise_parity_and_invalid_policy(self):
        from experiments.ghost_clipping_selection.render import CALCULATOR
        profiles = [Budget(),replace(Budget(),tokens=4096),
                    replace(Budget(),logical_examples=513,noise_multiplier=0,expected_batch_size=640)]
        source = CALCULATOR.split("const form=")[0]
        script = source+"\nconsole.log(JSON.stringify("+json.dumps([estimate(p)["input"] for p in profiles])+".map(estimateBudget)));"
        actual = json.loads(subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)
        for profile,result in zip(profiles,actual):
            for key,value in result.items():
                if value is None:
                    self.assertIsNone(estimate(profile)[key])
                else:
                    self.assertAlmostEqual(value,estimate(profile)[key],places=9)
        invalid = {**estimate(Budget())["input"],"clip":0}
        script = source+"\ntry{estimateBudget("+json.dumps(invalid)+");process.exit(1)}catch(e){console.log(e.message)}"
        self.assertIn("Clip",subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)


if __name__ == "__main__":
    unittest.main()
