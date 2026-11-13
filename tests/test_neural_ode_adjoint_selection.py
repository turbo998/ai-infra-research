from dataclasses import replace
import json
import math
import shutil
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.neural_ode_adjoint_selection.run import (
    Budget, Problem, checkpoint_reverse, continuous_backsolve_euler,
    continuous_exact, discrete_reverse, estimate, finite_difference, forward,
    load_latency, outputs, simulate,
)


class NeuralOdeAdjointTests(unittest.TestCase):
    def test_discrete_reverse_matches_independent_polynomial_and_finite_differences(self):
        profiles = [Problem(),replace(Problem(),theta=0),replace(Problem(),initial=0),
                    Problem(theta=-1.2,initial=2,target=-0.3,duration=0.5,steps=7),
                    Problem(theta=-4,steps=4)]
        for p in profiles:
            result = discrete_reverse(p)
            factor = 1+p.duration*p.theta/p.steps
            terminal = p.initial*factor**p.steps
            analytic_theta = (terminal-p.target)*p.initial*p.duration*factor**(p.steps-1)
            analytic_initial = (terminal-p.target)*factor**p.steps
            self.assertAlmostEqual(result["theta_gradient"],analytic_theta,places=11)
            self.assertAlmostEqual(result["initial_gradient"],analytic_initial,places=11)
            self.assertAlmostEqual(result["theta_gradient"],finite_difference(p,"theta"),places=7)
            self.assertAlmostEqual(result["initial_gradient"],finite_difference(p,"initial"),places=7)

    def test_checkpoint_replay_preserves_all_block_sizes_including_partial_last_block(self):
        p = replace(Problem(),steps=7)
        reference = discrete_reverse(p)
        for block in range(1,8):
            replay = checkpoint_reverse(p,block)
            for key in ("terminal","loss","theta_gradient","initial_gradient"):
                self.assertEqual(replay[key],reference[key])
            self.assertEqual(replay["replay_evaluations"],7)
            self.assertEqual(replay["checkpoint_slots"],math.ceil(7/block)+1)
            self.assertEqual(replay["maximum_local_replay_slots"],block+1)

    def test_continuous_and_discrete_objective_sign_difference_is_explicit(self):
        p = Problem()
        discrete,continuous = discrete_reverse(p),continuous_exact(p)
        self.assertLess(discrete["terminal"],p.target)
        self.assertGreater(continuous["terminal"],p.target)
        self.assertLess(discrete["theta_gradient"],0)
        self.assertGreater(continuous["theta_gradient"],0)
        self.assertAlmostEqual(continuous["theta_gradient"],
                               (math.exp(0.7)-2)*math.exp(0.7),places=12)
        rows = simulate()["convergence"]
        self.assertTrue(all(b["absolute_gradient_difference"] < a["absolute_gradient_difference"]
                            for a,b in zip(rows,rows[1:])))
        self.assertGreater(rows[-1]["discrete_gradient"],0)

    def test_backward_time_explicit_euler_state_resets_do_not_make_discrete_reverse(self):
        p = Problem()
        plain,reset = continuous_backsolve_euler(p),continuous_backsolve_euler(p,1)
        reference = discrete_reverse(p)
        expected_initial = reference["terminal"]*(1-p.theta*p.duration/p.steps)**p.steps
        self.assertAlmostEqual(plain["reconstructed_initial_before_final_reset"],expected_initial)
        self.assertLess(reset["maximum_state_difference_before_reset"],plain["maximum_state_difference_before_reset"])
        self.assertNotAlmostEqual(reset["theta_gradient"],reference["theta_gradient"],places=4)
        self.assertGreater(abs(reset["theta_gradient"]-reference["theta_gradient"]),
                           abs(plain["theta_gradient"]-reference["theta_gradient"]))
        self.assertAlmostEqual(reset["theta_gradient"],
                               (reference["terminal"]-p.target)*reference["terminal"]*p.duration)

    def test_selected_state_budget_observations_and_non_saving_extreme_blocks(self):
        result = estimate(Budget())
        self.assertEqual(result["full_tape_state_bytes"],4198498304)
        self.assertEqual(result["block_selected_state_bytes"],276824064)
        self.assertEqual(result["block_checkpoint_slots"],33)
        self.assertEqual(result["maximum_replay_slots"],33)
        self.assertEqual(result["saved_observation_state_bytes"],8388608)
        self.assertEqual(result["augmented_y_adjoint_parameter_bytes"],12582912)
        for block in (1,1000):
            extreme = estimate(replace(Budget(),block=block))
            self.assertGreater(extreme["block_selected_state_bytes"],extreme["full_tape_state_bytes"])
        self.assertIsNone(result["peak_memory_bytes"])

    def test_invalid_problem_replay_and_budget_are_explicit_errors(self):
        for p in (replace(Problem(),steps=0),replace(Problem(),steps=True),
                  replace(Problem(),theta=math.nan),replace(Problem(),duration=0),
                  replace(Problem(),theta=101)):
            with self.assertRaises(ValueError):
                forward(p)
        for block in (0,5,True):
            with self.assertRaises(ValueError):
                checkpoint_reverse(Problem(),block)
        for profile in (replace(Budget(),block=1001),replace(Budget(),element_bytes=3),
                        replace(Budget(),observation_times=1),
                        replace(Budget(),state_elements=9007199254740991)):
            with self.assertRaises(ValueError):
                estimate(profile)

    def test_sealed_timing_medians_artifacts_csp_and_solid_geometry(self):
        for row in load_latency()["rows"]:
            for key in ("full_tape","block_replay"):
                self.assertEqual(row[key+"_median_ms"],statistics.median(row["samples_ns"][key])/1e6)
        for path,content in outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
            if path.name.endswith("-cards.html"):
                self.assertEqual(content.count('<section class="card"'),7)
                self.assertNotIn("<script",content)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'",content)
                self.assertNotIn('style="width:-',content)
            if path.suffix == ".svg":
                root = ET.fromstring(content)
                self.assertEqual(root.attrib["viewBox"],"0 0 1600 1000")
                for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
                    self.assertTrue(all(200 <= float(point.split(",")[1]) <= 620
                                        for point in polygon.attrib["points"].split()))

    @unittest.skipUnless(shutil.which("node"),"Node.js optional")
    def test_javascript_checkpoint_ceil_observation_and_parameter_budget_parity(self):
        from experiments.neural_ode_adjoint_selection.render import CALCULATOR
        profiles = [Budget(),replace(Budget(),steps=7,block=3),
                    replace(Budget(),block=1000),replace(Budget(),observation_times=100,parameter_elements=10)]
        source = CALCULATOR.split("const form=")[0]
        script = source+"\nconsole.log(JSON.stringify("+json.dumps([estimate(p)["input"] for p in profiles])+".map(estimateBudget)));"
        actual = json.loads(subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)
        for profile,result in zip(profiles,actual):
            for key,value in result.items():
                self.assertEqual(value,estimate(profile)[key])
        invalid = {**estimate(Budget())["input"],"block":1001}
        script = source+"\ntry{estimateBudget("+json.dumps(invalid)+");process.exit(1)}catch(e){console.log(e.message)}"
        self.assertIn("Block",subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)


if __name__ == "__main__":
    unittest.main()
