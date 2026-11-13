from dataclasses import replace
import json
import math
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.gnn_neighbor_sampling_selection.run import (
    Budget, FEATURES, GRAPH, LABELS, SEEDS, estimate, feature_gradient, forward,
    nonlinear_expectations, outputs, sample, seed_loss, simulate,
)


class GnnNeighborSamplingTests(unittest.TestCase):
    def test_full_neighbor_closure_preserves_seed_outputs_and_global_ids(self):
        metadata=sample(GRAPH,SEEDS,[-1,-1])
        self.assertEqual(metadata["n_id"][:2],SEEDS)
        local_x=[FEATURES[node] for node in metadata["n_id"]]
        local,_=forward(metadata["local_graph"],local_x)
        full,_=forward(GRAPH,FEATURES)
        self.assertEqual(local[-1][:2],[full[-1][node] for node in SEEDS])
        self.assertEqual(len(set(metadata["n_id"])),len(metadata["n_id"]))
        for node,neighbors in zip(metadata["n_id"],metadata["local_graph"]):
            self.assertTrue(all(metadata["n_id"][j] in GRAPH[node] for j in neighbors))

    def test_seed_only_feature_gradients_match_full_graph_and_finite_differences(self):
        expected=feature_gradient(GRAPH,FEATURES,SEEDS,LABELS)
        metadata=sample(GRAPH,SEEDS,[-1,-1])
        local=feature_gradient(metadata["local_graph"],[FEATURES[n] for n in metadata["n_id"]],[0,1],LABELS)
        for index,node in enumerate(metadata["n_id"]):
            for value,reference in zip(local[index],expected[node]):
                self.assertAlmostEqual(value,reference,places=12)
        epsilon=1e-5
        for node in range(len(GRAPH)):
            for column in range(2):
                plus,minus=[row[:] for row in FEATURES],[row[:] for row in FEATURES]
                plus[node][column]+=epsilon
                minus[node][column]-=epsilon
                numerical=(seed_loss(forward(GRAPH,plus)[0][-1],SEEDS,LABELS)
                           -seed_loss(forward(GRAPH,minus)[0][-1],SEEDS,LABELS))/(2*epsilon)
                self.assertAlmostEqual(expected[node][column],numerical,places=9)

    def test_missing_hop_mapping_direction_and_degree_controls(self):
        cases={row["id"]:row for row in simulate()["cases"]}
        self.assertTrue(cases["full-neighbor-closure"]["matches_reference"])
        for name in ("one-hop-for-two-layers","wrong-local-feature-index","reverse-message-direction","sample-mean-with-full-degree"):
            self.assertFalse(cases[name]["matches_reference"])
            self.assertGreater(cases[name]["maximum_absolute_error"],0.1)
        sampled=cases["finite-fanout-first-neighbor"]
        self.assertTrue(sampled["intentional_sampling_approximation"])
        self.assertFalse(sampled["matches_reference"])
        self.assertIn("same-sampled",cases["sample-mean-with-full-degree"]["reference_scope"])

    def test_exact_uniform_enumeration_separates_linear_and_nonlinear_expectations(self):
        rows=nonlinear_expectations()
        self.assertEqual([row["equally_likely_subsets"] for row in rows],[3,3,1])
        for row in rows:
            self.assertEqual(row["expected_sample_mean"],row["full_linear_mean"])
        self.assertAlmostEqual(rows[0]["expected_relu_of_sample_mean"],2/3)
        self.assertAlmostEqual(rows[1]["expected_relu_of_sample_mean"],1/3)
        self.assertEqual(rows[2]["expected_relu_of_sample_mean"],0)

    def test_fanout_occurrence_bound_caps_unique_without_fabricating_count(self):
        result=estimate(Budget())
        self.assertEqual(result["tree_occurrence_bound"],1024*(1+15+150))
        self.assertEqual(result["feature_subset_bound_bytes"],169984*128*2)
        self.assertIsNone(result["actual_unique_nodes"])
        capped=estimate(replace(Budget(),graph_nodes=2000))
        self.assertEqual(capped["unique_node_upper_bound"],2000)
        self.assertEqual(capped["subset_transfer_modeled_ms"],capped["full_transfer_modeled_ms"])
        self.assertLess(result["subset_transfer_modeled_ms"],result["full_transfer_modeled_ms"])

    def test_isolated_seed_and_invalid_inputs_are_explicit(self):
        metadata=sample(GRAPH,[6],[-1,-1])
        self.assertEqual(metadata["n_id"],[6])
        values,_=forward(metadata["local_graph"],[FEATURES[6]])
        self.assertTrue(all(math.isfinite(value) for value in values[-1][0]))
        for profile in (replace(Budget(),fanouts=(0,10)),replace(Budget(),seeds=1000001),
                        replace(Budget(),bandwidth_gbps=math.nan)):
            with self.assertRaises(ValueError):
                estimate(profile)
        with self.assertRaises(ValueError):
            sample(GRAPH,[4,4],[-1])
        with self.assertRaises(ValueError):
            forward([[7]],[FEATURES[0]])

    def test_generated_seven_cards_csp_and_svg_geometry(self):
        for path,content in outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
            if path.name.endswith("-cards.html"):
                self.assertEqual(content.count('<section class="card"'),7)
                self.assertNotIn("<script",content)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'",content)
            if path.suffix==".svg":
                root=ET.fromstring(content)
                self.assertEqual(root.attrib["viewBox"],"0 0 1600 1000")
                for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
                    for point in polygon.attrib["points"].split():
                        self.assertTrue(200<=float(point.split(",")[1])<=620)

    @unittest.skipUnless(shutil.which("node"),"Node.js optional")
    def test_javascript_budget_parity_and_invalid_fanout(self):
        from experiments.gnn_neighbor_sampling_selection.render import CALCULATOR
        profiles=[Budget(),replace(Budget(),graph_nodes=2000),replace(Budget(),fanouts=(5,5,2))]
        source=CALCULATOR.split("const form=")[0]
        script=source+"\nconsole.log(JSON.stringify("+json.dumps([estimate(p)["input"] for p in profiles])+".map(estimateBudget)));"
        actual=json.loads(subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)
        for profile,result in zip(profiles,actual):
            for key,value in result.items():
                if value is None:
                    self.assertIsNone(estimate(profile)[key])
                else:
                    self.assertAlmostEqual(value,estimate(profile)[key],places=9)
        invalid=source+"\ntry{estimateBudget("+json.dumps({**estimate(Budget())["input"],"fanouts":[15,0]})+");process.exit(1)}catch(e){console.log(e.message)}"
        result=subprocess.run(["node","-e",invalid],check=True,capture_output=True,text=True)
        self.assertIn("fanouts",result.stdout)


if __name__=="__main__":
    unittest.main()
