import json
import random
import shutil
import subprocess
import unittest

from experiments.grouped_moe_padding_mask_selection import run as m
from experiments.grouped_moe_padding_mask_selection.render import JS


class GroupedPaddingTests(unittest.TestCase):
    def test_default_grouped_live_outputs_and_unchanged_shape(self):
        before, after = m.route(m.DEFAULT, True), m.route(m.DEFAULT)
        self.assertEqual((before["original_scalar_expert_calls"], after["original_scalar_expert_calls"]), (16,8))
        self.assertEqual(before["logical_int32_fp32_route_bytes"],128)
        self.assertEqual(after["logical_int32_fp32_route_bytes"],128)
        for i,pad in enumerate(m.DEFAULT["padding"]):
            if pad:
                self.assertEqual((after["ids"][i],after["weight_numerators"][i],after["output_numerators"][i]),([-1,-1],[0,0],0))
            else:
                self.assertEqual(before["output_numerators"][i],after["output_numerators"][i])
        self.assertEqual(m.DEFAULT["ids"][2],[0,2])

    def test_aiter_and_ungrouped_earlyreturn(self):
        self.assertFalse(m.route({**m.DEFAULT,"aiter_fused":True})["mask_applied"])
        self.assertTrue(m.route({**m.DEFAULT,"aiter_fused":True,"model_optin":True})["mask_applied"])
        r=m.route({**m.DEFAULT,"groups":4})
        self.assertTrue(r["router_masks_padding_property"])
        self.assertFalse(r["mask_applied"])
        self.assertEqual(r["reason"],"ungrouped_early_return")

    def test_environment_context_and_witness_are_separate(self):
        for change in ({"env_skip":False},{"context":False},{"padding":None}):
            r=m.route({**m.DEFAULT,**change})
            self.assertTrue(r["router_masks_padding_property"])
            self.assertFalse(r["mask_applied"])
            self.assertEqual(r["original_scalar_expert_calls"],16)

    def test_short_mask_is_not_partial_global_mask(self):
        p={**m.DEFAULT,"model_optin":True,"padding":[False,True]}
        r=m.route(p)
        self.assertEqual(r["reason"],"mask_undercoverage_return_none")
        self.assertFalse(r["mask_applied"])
        self.assertEqual(r["ids"],m.DEFAULT["ids"])
        with self.assertRaises(m.LegacyCoverageError):
            m.route(p,True)
        self.assertEqual(m.route({**m.DEFAULT,"padding":m.DEFAULT["padding"]+[True]*8})["ids"],
                         m.route(m.DEFAULT)["ids"])

    def test_sentinel_requires_consumer_and_zero_weight(self):
        with self.assertRaisesRegex(ValueError,"consumer"):
            m.route({**m.DEFAULT,"consumer_skip_sentinel":False})
        with self.assertRaisesRegex(ValueError,"zero"):
            m.dispatch([1],[[-1]],[[1]],[2])
        with self.assertRaises(ValueError):
            m.dispatch([1],[[-2]],[[0]],[2])
        self.assertEqual(m.dispatch([3],[[-1]],[[0]],[2]),([0],0))
        self.assertEqual(m.dispatch([3],[[0]],[[0]],[2]),([0],1))

    def test_explicit_schema_errors_and_empty_rows(self):
        for change in ({"topk":True},{"groups":0},{"env_skip":1},{"padding":[0]},
                       {"ids":[[-1,2]]*8},{"expert_coefficients":[2]},{"extra":True}):
            with self.assertRaises(ValueError):
                m.route({**m.DEFAULT,**change})
        r=m.route({**m.DEFAULT,"values":[],"ids":[],"weights":[],"padding":[]})
        self.assertEqual((r["output_numerators"],r["original_scalar_expert_calls"]),([],0))

    def test_zero_weight_timing_baseline_preserves_all_outputs(self):
        for n in (64,256,1024,4096):
            a,b=m.paths(n);x,y=a(),b()
            self.assertEqual(x[0],y[0])
            self.assertEqual((x[1],y[1]),(n,2*n))

    @unittest.skipUnless(shutil.which("node"),"Node required for original H5policy parity")
    def test_random_live_invariance_and_node_parity(self):
        rng=random.Random(159)
        cases=[m.DEFAULT]
        for _ in range(64):
            pad=[bool(rng.randrange(2)) for _ in range(8)]
            p={**m.DEFAULT,"padding":pad,"aiter_fused":bool(rng.randrange(2)),
               "model_optin":bool(rng.randrange(2)),"env_skip":bool(rng.randrange(2)),
               "context":bool(rng.randrange(2)),"groups":rng.choice([1,2,4])}
            a,b=m.route(p,True),m.route(p)
            for i,v in enumerate(pad):
                if not v:
                    self.assertEqual(a["output_numerators"][i],b["output_numerators"][i])
            cases.append(p)
        script=JS+"\nconst p=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(p.map(v=>[route(v,true),route(v)])));"
        result=subprocess.run(["node","-e",script],input=json.dumps(cases),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout),[[m.route(v,True),m.route(v)] for v in cases])


if __name__=="__main__":
    unittest.main()
