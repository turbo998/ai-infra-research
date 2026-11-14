import copy
import json
import random
import subprocess
import unittest

from experiments.optional_dbias_allocation_selection import run as m
from experiments.optional_dbias_allocation_selection.render import JS,cards


class OptionalDbiasTests(unittest.TestCase):
    def test_independent_long_context_bytes(self):
        r=m.evaluate(m.DEFAULT)
        self.assertEqual(r["placeholder_shape"],[4,32,16384,16384])
        self.assertEqual(int(r["dense_placeholder_bytes"]),64*2**30)
        self.assertEqual((int(r["old_selected_live_bytes"]),int(r["new_selected_live_bytes"])),(70*2**30,6*2**30))
        self.assertFalse(r["old_selected_fits_budget"]);self.assertTrue(r["new_selected_fits_budget"])
        self.assertEqual(int(r["ragged_rectangular_pair_positions"]),16384**2+2048*1024+512**2+128**2)
        self.assertIsNone(r["native_oom_prediction"])

    def test_requested_gradient_preserved_and_missing_rejected(self):
        p={**m.DEFAULT,"bias_requires_grad":True,"grad_bias_provided":True,"provided_grad_bias_bytes":str(64*2**30)}
        r=m.evaluate(p)
        self.assertEqual(r["new_dbias_return"],"provided_grad_bias");self.assertTrue(r["kernel_has_dbias"])
        self.assertEqual(r["removed_unused_bytes"],"0")
        self.assertEqual(r["new_selected_live_bytes"],str(70*2**30))
        p["grad_bias_provided"]=False;p["provided_grad_bias_bytes"]=None
        with self.assertRaises(ValueError):
            m.evaluate(p)

    def test_provided_unrequested_buffer_is_not_freed(self):
        r=m.evaluate({**m.DEFAULT,"grad_bias_provided":True,"provided_grad_bias_bytes":str(64*2**30)})
        self.assertEqual(r["new_dbias_return"],"undefined_tensor")
        self.assertEqual(r["provided_caller_owned_bytes"],str(64*2**30))
        self.assertEqual(r["new_selected_live_bytes"],str(70*2**30));self.assertFalse(r["new_selected_fits_budget"])

    def test_causal_gqa_and_empty_state(self):
        for change in ({"causal":False},{"kv_heads":4}):
            self.assertEqual(m.evaluate({**m.DEFAULT,**change})["dense_placeholder_bytes"],str(64*2**30))
        r=m.evaluate({**m.DEFAULT,"q_lengths":[],"k_lengths":[],"other_live_bytes":"0"})
        self.assertEqual(r["dense_placeholder_bytes"],"0");self.assertEqual(r["new_dbias_return"],"undefined_tensor")
        self.assertEqual(r["new_fresh_return_region_bytes"],"0")

    def test_exact_counts_above_javascript_number_range(self):
        p={**m.DEFAULT,"q_lengths":[2**20]*64,"k_lengths":[2**20]*64,"query_heads":128,"kv_heads":1}
        self.assertEqual(m.evaluate(p)["dense_placeholder_bytes"],str(2**54))

    def test_invalid_shape_flags_metadata(self):
        changes=[{"layout":"fixed"},{"q_lengths":[True]},{"kv_heads":3},{"bias_requires_grad":1},
                 {"dtype":"fp32"},{"budget_bytes":8},{"other_live_bytes":"-1"},{"other_live_bytes":"١"},{"provided_grad_bias_bytes":"4"}]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.evaluate({**m.DEFAULT,**change})

    def test_independent_small_pair_oracle_and_node_parity(self):
        rng=random.Random(151);inputs=[p for _,p in m.scenarios()]
        for _ in range(64):
            qs=[rng.randrange(6) for _ in range(4)];ks=[rng.randrange(6) for _ in range(4)]
            p={**m.DEFAULT,"q_lengths":qs,"k_lengths":ks}
            expected=sum(1 for q,k in zip(qs,ks) for _ in range(q) for _ in range(k))
            self.assertEqual(int(m.evaluate(p)["ragged_rectangular_pair_positions"]),expected)
            inputs.append(p)
        inputs.append({**m.DEFAULT,"q_lengths":[2**20]*64,"k_lengths":[2**20]*64,"query_heads":128,"kv_heads":1})
        output=subprocess.run(["node"],input=JS+"\nconsole.log(JSON.stringify("+json.dumps(inputs)+".map(evaluate)));",
                              capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(output.stdout),[m.evaluate(p) for p in inputs])

    def test_sealed_full_policy_matrix_and_no_native_claims(self):
        latency=json.loads((m.HERE/"latency.json").read_text())
        sealed=json.loads((m.HERE/"report.json").read_text())
        self.assertEqual(sealed,m.build_report(latency));self.assertFalse(sealed["independently_reproduced"])
        self.assertIsNone(sealed["native_allocator_peak_bytes"])
        html=cards(sealed);self.assertEqual(html.count('class="card"'),7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",html);self.assertNotIn("<script",html)
        for row in sealed["matrix"]:
            self.assertIn(row["scenario"],html)
