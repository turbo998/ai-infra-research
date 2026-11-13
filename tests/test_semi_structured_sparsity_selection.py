from dataclasses import replace
import itertools
import json
import math
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.semi_structured_sparsity_selection.run import (
    Budget, RHS, WEIGHTS, estimate, matmul, maximum_error, outputs, pack,
    pattern_valid, prune_bidirectional, prune_rows, simulate, sparse_matmul,
    transpose, unpack,
)


class SemiStructuredSparsityTests(unittest.TestCase):
    def test_explicit_row_pruning_and_compression_are_distinct(self):
        pruned = prune_rows(WEIGHTS)
        self.assertTrue(pattern_valid(pruned))
        self.assertFalse(pattern_valid(WEIGHTS))
        with self.assertRaises(ValueError):
            pack(WEIGHTS)
        self.assertEqual(prune_rows([[1,-1,1,-1]]),[[1,-1,0,0]])
        self.assertEqual(prune_rows([[0,0,0,0]]),[[0,0,0,0]])
        with self.assertRaises(ValueError):
            prune_rows([[1,2,3]])

    def test_all_six_logical_codes_and_zero_filler_slots_round_trip(self):
        for i,j in itertools.combinations(range(4),2):
            row = [0.0]*4
            row[i],row[j] = 2.5,-1.25
            encoded = pack([row])
            self.assertEqual(encoded["logical_nibbles"],[[i | (j << 2)]])
            self.assertEqual(unpack(encoded),[row])
        for row in ([0,0,0,0],[0,0,3,0]):
            self.assertEqual(unpack(pack([row])),[row])
        bad = pack([[1,2,0,0]])
        bad["logical_nibbles"][0][0] = 0
        with self.assertRaises(ValueError):
            unpack(bad)

    def test_sparse_product_and_dense_backward_match_pruned_reference(self):
        pruned = prune_rows(WEIGHTS)
        reference = matmul(pruned,RHS)
        self.assertLess(maximum_error(sparse_matmul(pack(pruned),RHS),reference),1e-12)
        analytic = matmul(transpose(pruned),reference)
        epsilon = 1e-5
        for i in range(8):
            for j in range(3):
                plus,minus = [row[:] for row in RHS],[row[:] for row in RHS]
                plus[i][j] += epsilon
                minus[i][j] -= epsilon
                loss = lambda x:0.5*math.fsum(v*v for row in matmul(pruned,x) for v in row)
                self.assertAlmostEqual((loss(plus)-loss(minus))/(2*epsilon),analytic[i][j],places=6)
        cases = {row["id"]:row for row in simulate()["cases"]}
        self.assertGreater(cases["reverse-metadata-index-order"]["maximum_absolute_error"],6)
        self.assertTrue(cases["pruned-vs-original-dense"]["intentional_pruning"])

    def test_forward_pattern_is_not_transpose_pattern_and_joint_policy_is_explicit(self):
        pruned = prune_rows(WEIGHTS)
        self.assertFalse(pattern_valid(transpose(pruned)))
        with self.assertRaises(ValueError):
            pack(transpose(pruned))
        joint = prune_bidirectional(WEIGHTS)
        self.assertTrue(pattern_valid(joint))
        self.assertTrue(pattern_valid(transpose(joint)))
        self.assertNotEqual(joint,pruned)
        self.assertEqual(unpack(pack(transpose(joint))),transpose(joint))

    def test_metadata_orientation_padding_and_original_weight_ratio(self):
        b = estimate(Budget())
        self.assertEqual(b["logical_metadata_bytes"],4096*4096//8)
        self.assertEqual(b["compressed_to_original_ratio"],9/16)
        self.assertGreater(b["both_orientation_logical_bytes"],b["original_dense_weight_bytes"])
        odd = estimate(replace(Budget(),output_features=33,input_features=65,tokens=1))
        self.assertEqual(odd["forward_padded_shape"],[64,128])
        self.assertEqual(odd["transpose_padded_shape"],[96,64])
        self.assertEqual(odd["padded_tokens"],8)
        self.assertFalse(odd["source_shape_gate_passes_without_padding"])
        other = estimate(replace(Budget(),backend="cusparselt",output_features=33,input_features=65))
        self.assertEqual(other["forward_padded_shape"],[48,80])
        self.assertEqual(other["transpose_padded_shape"],[80,48])

    def test_strict_amortization_threshold_and_no_saving_boundary(self):
        b = Budget()
        result = estimate(b)
        self.assertEqual(result["minimum_reuse_for_strict_saving"],63)
        lower = estimate(replace(b,reuse=62))
        upper = estimate(replace(b,reuse=63))
        self.assertGreaterEqual(lower["sparse_modeled_total_ms"],lower["dense_modeled_total_ms"])
        self.assertLess(upper["sparse_modeled_total_ms"],upper["dense_modeled_total_ms"])
        self.assertIsNone(estimate(replace(b,tokens=1))["minimum_reuse_for_strict_saving"])
        self.assertIsNone(estimate(replace(b,sparse_effective_tflops=500))["minimum_reuse_for_strict_saving"])
        self.assertEqual(estimate(replace(b,pack_ms=0))["minimum_reuse_for_strict_saving"],1)
        for profile in (replace(b,backend="unknown"),replace(b,reuse=0),replace(b,pack_ms=-1),
                        replace(b,dense_effective_tflops=math.nan)):
            with self.assertRaises(ValueError):
                estimate(profile)

    def test_reproducible_cards_csp_and_geometry(self):
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
    def test_javascript_backend_padding_and_reuse_parity(self):
        from experiments.semi_structured_sparsity_selection.render import CALCULATOR
        profiles = [Budget(),replace(Budget(),tokens=1),
                    replace(Budget(),backend="cusparselt",output_features=33,input_features=65),
                    replace(Budget(),pack_ms=0)]
        source = CALCULATOR.split("const form=")[0]
        script = source+"\nconsole.log(JSON.stringify("+json.dumps([estimate(p)["input"] for p in profiles])+".map(estimateBudget)));"
        actual = json.loads(subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)
        for profile,result in zip(profiles,actual):
            for key,value in result.items():
                if isinstance(value,list) or value is None or isinstance(value,bool):
                    self.assertEqual(value,estimate(profile)[key])
                else:
                    self.assertAlmostEqual(value,estimate(profile)[key],places=9)
        invalid = {**estimate(Budget())["input"],"backend":"unknown"}
        script = source+"\ntry{estimateBudget("+json.dumps(invalid)+");process.exit(1)}catch(e){console.log(e.message)}"
        self.assertIn("backend",subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)


if __name__ == "__main__":
    unittest.main()
