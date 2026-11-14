import copy
import json
import random
import subprocess
import unittest
from experiments.xcd_tail_placement_selection.run import DEFAULT,enumerate_ownership,ownership,plan
from experiments.xcd_tail_placement_selection.render import JS


class XCDTailTests(unittest.TestCase):
    def test_ten_request_capacity_and_padding(self):
        a=plan(DEFAULT);b=plan(DEFAULT,"request_only")
        self.assertEqual((a["selected_splits"],a["placement"]),(3,"balanced"))
        self.assertEqual(a["valid_rows_per_chiplet"],[10]*8)
        self.assertEqual((a["launched_workgroups"],a["padding_workgroups"]),(240,0))
        self.assertEqual((b["selected_splits"],b["launched_workgroups"],b["padding_workgroups"]),(2,256,96))

    def test_one_round_ranges_and_locality_ties(self):
        for n,s,kind in ((1,4,"request"),(8,4,"request"),(9,3,"balanced"),(10,3,"balanced"),
                         (11,2,"request"),(16,2,"request"),(17,1,"request"),(24,1,"request"),(32,1,"request")):
            a=plan({**DEFAULT,"requests":n})
            self.assertEqual((a["selected_splits"],a["placement"]),(s,kind))

    def test_outside_scope_is_not_a_zero_split_plan(self):
        for n,followup in ((33,"occupancy_2_candidate"),(49,"plain_cost_model")):
            a=plan({**DEFAULT,"requests":n})
            self.assertIsNone(a["selected_splits"])
            self.assertEqual(a["native_followup"],followup)
        for change in ({"topk":32},{"query_rows":4},{"kv_heads":2},{"chiplets":0},{"chiplets":1}):
            self.assertEqual(plan({**DEFAULT,**change})["status"],"outside_tuned_profile")

    def test_explicit_override_not_automatic_capacity_claim(self):
        a=plan({**DEFAULT,"explicit_splits":4})
        self.assertEqual((a["placement"],a["max_valid_workgroups_per_chiplet"]),("request",64))
        self.assertFalse(a["one_round"])

    def test_row_mapping_bijection_and_independent_count(self):
        rng=random.Random(170)
        for _ in range(64):
            n=rng.randrange(1,65);q=rng.randrange(1,17);x=rng.randrange(1,17)
            for kind in ("request","balanced"):
                counts,grid,mapped=enumerate_ownership(n,q,x,kind)
                self.assertEqual((counts,grid),ownership(n,q,x,kind))
                self.assertEqual(sorted(mapped),list(range(grid)))
                self.assertEqual(sum(counts),n*q)

    def test_no_work_and_unknown_workspace(self):
        a=plan({**DEFAULT,"requests":0})
        self.assertEqual(a["launched_workgroups"],0)
        self.assertIsNone(a["native_workspace_bytes"])
        self.assertIsNone(plan(DEFAULT)["native_gpu_verified"])

    def test_invalid_topology_explicit_errors(self):
        for change in ({"requests":True},{"cus":255},{"explicit_splits":0},{"query_rows":0},{"chiplets":17}):
            with self.subTest(change=change),self.assertRaises(ValueError):plan({**DEFAULT,**change})

    def test_node_parity65profiles(self):
        rng=random.Random(170);profiles=[copy.deepcopy(DEFAULT)]
        for _ in range(64):
            x=rng.choice([1,2,4,8,16])
            profiles.append({**DEFAULT,"requests":rng.randrange(65),"chiplets":x,"cus":x*rng.choice([16,32,64]),
                             "query_rows":rng.choice([4,8]),"topk":rng.choice([16,32]),"kv_heads":rng.choice([1,2]),
                             "explicit_splits":rng.choice([None,1,2,4])})
        expected=[[plan(p,s) for s in ("compare","request_only")] for p in profiles]
        script=JS+"\nconst ps="+json.dumps(profiles)+";console.log(JSON.stringify(ps.map(p=>['compare','request_only'].map(s=>plan(p,s)))));"
        out=subprocess.run(["node"],input=script,check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(out.stdout),expected)
