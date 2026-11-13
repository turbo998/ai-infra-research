from dataclasses import replace
import itertools
import json
import math
import shutil
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.set_prediction_matching_selection.run import (
    Budget, COST, PREDICTIONS, TARGETS, assignment_cost, distributed_gradient,
    enumerate_assignment, estimate, greedy_assignment, load_latency, outputs,
    scalar_loss, simulate, subset_assignment,
)


class SetPredictionMatchingTests(unittest.TestCase):
    def test_independent_rectangular_solvers_match_with_empty_negative_and_tied_costs(self):
        profiles = [COST,[[],[],[]],[[0,0],[0,0],[0,0]],[[-2,1],[0,-3],[4,2]]]
        for targets in range(1,5):
            profiles.append([[((q+3)*(t+5)%17)+0.01*q for t in range(targets)]
                             for q in range(targets+2)])
        for matrix in profiles:
            exact,dp = enumerate_assignment(matrix),subset_assignment(matrix)
            self.assertAlmostEqual(dp["cost"],exact["cost"],places=12)
            self.assertAlmostEqual(assignment_cost(matrix,dp["query_for_target"]),exact["cost"],places=12)
        self.assertEqual(subset_assignment([[0,0],[0,0]])["query_for_target"],(0,1))

    def test_greedy_can_be_suboptimal_and_duplicate_argmin_is_infeasible(self):
        self.assertEqual(enumerate_assignment(COST),{"cost":4,"query_for_target":(1,0)})
        self.assertEqual(greedy_assignment(COST)["cost"],81)
        with self.assertRaises(ValueError):
            assignment_cost(COST,(0,0))
        for matrix in ([],[[1],[1,2]],[[1,2]],[[math.nan]],[[math.inf]]):
            with self.assertRaises(ValueError):
                subset_assignment(matrix)

    def test_target_and_query_permutations_preserve_supervision_with_fresh_identity(self):
        reference = scalar_loss(PREDICTIONS,TARGETS)
        for order in itertools.permutations(range(3)):
            actual = scalar_loss([PREDICTIONS[i] for i in order],list(reversed(TARGETS)))
            restored = [0.0]*3
            for new,old in enumerate(order):
                restored[old] = actual["gradient"][new]
            self.assertEqual(restored,reference["gradient"])
            self.assertEqual(actual["loss"],reference["loss"])
        report = simulate()["scalar_supervision"]
        self.assertEqual(report["stale_reordered"]["loss"],41)
        self.assertEqual(report["stale_gradient_restored"],[-1,0,9])
        self.assertEqual(scalar_loss(PREDICTIONS,[])["gradient"],[0,0,0])

    def test_detached_unique_region_gradient_and_cost_transform_objective_boundary(self):
        report = simulate()
        supervision = report["scalar_supervision"]
        self.assertLess(supervision["finite_difference_error"],1e-8)
        self.assertEqual(supervision["baseline"]["gradient"],[0,1,0])
        change = report["cost_objective_switch"]
        self.assertEqual(change["negative_probability_assignment"]["query_for_target"],(0,1))
        self.assertEqual(change["negative_log_assignment"]["query_for_target"],(1,0))
        self.assertTrue(change["intentional_objective_change"])

    def test_ddp_global_count_compensation_empty_rank_and_low_count_clamp(self):
        result = distributed_gradient([1,3,0],[2,18,0])
        self.assertEqual(result["ddp_average_gradient"],5)
        self.assertEqual(distributed_gradient([4],[20])["ddp_average_gradient"],5)
        self.assertEqual(distributed_gradient([2,2],[4,16])["ddp_average_gradient"],5)
        self.assertAlmostEqual(distributed_gradient([1,3,0],[2,18,0],"rank_local")["ddp_average_gradient"],8/3)
        self.assertAlmostEqual(distributed_gradient([1,3,0],[2,18,0],"global_without_world_compensation")["ddp_average_gradient"],5/3)
        low = distributed_gradient([1,0,0,0],[2,0,0,0])
        self.assertEqual(low["ddp_average_gradient"],0.5)
        self.assertEqual(low["ideal_object_mean"],2)
        self.assertEqual(distributed_gradient([0,0],[0,0])["ddp_average_gradient"],0)
        for counts,sums,policy in (([],[],"source_global"),([0],[1],"source_global"),
                                  ([True],[0],"source_global"),([1],[math.nan],"source_global"),
                                  ([1],[1],"unknown")):
            with self.assertRaises(ValueError):
                distributed_gradient(counts,sums,policy)

    def test_cross_image_array_budget_auxiliary_work_and_safe_integer_limits(self):
        result = estimate(Budget())
        self.assertEqual(result["flattened_cross_image_bytes"],3072000)
        self.assertEqual(result["per_image_block_bytes"],384000)
        self.assertEqual(result["main_and_aux_matching_passes"],6)
        self.assertEqual(result["flattened_cost_bytes_written_across_passes"],18432000)
        self.assertEqual(estimate(replace(Budget(),targets_per_image=0))["flattened_cross_image_bytes"],0)
        for profile in (replace(Budget(),queries=1),replace(Budget(),cost_bytes=2),
                        replace(Budget(),auxiliary_layers=-1),replace(Budget(),batch=True),
                        replace(Budget(),batch=9007199254740991),
                        replace(Budget(),targets_per_image=0,auxiliary_layers=9007199254740991)):
            with self.assertRaises(ValueError):
                estimate(profile)

    def test_sealed_cpu_samples_artifacts_script_free_cards_and_geometry(self):
        latency = load_latency()
        for row in latency["rows"]:
            for key in ("subset_dp","enumeration"):
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
    def test_javascript_batch_auxiliary_and_empty_target_formula_parity(self):
        from experiments.set_prediction_matching_selection.render import CALCULATOR
        profiles = [Budget(),replace(Budget(),batch=1,auxiliary_layers=0),
                    replace(Budget(),targets_per_image=0),replace(Budget(),cost_bytes=8)]
        source = CALCULATOR.split("const form=")[0]
        script = source+"\nconsole.log(JSON.stringify("+json.dumps([estimate(p)["input"] for p in profiles])+".map(estimateBudget)));"
        actual = json.loads(subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)
        for profile,result in zip(profiles,actual):
            for key,value in result.items():
                self.assertEqual(value,estimate(profile)[key])
        invalid = {**estimate(Budget())["input"],"queries":1}
        script = source+"\ntry{estimateBudget("+json.dumps(invalid)+");process.exit(1)}catch(e){console.log(e.message)}"
        self.assertIn("queries",subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)


if __name__ == "__main__":
    unittest.main()
