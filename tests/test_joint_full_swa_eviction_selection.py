import copy
import json
import random
import subprocess
import unittest
from experiments.joint_full_swa_eviction_selection.run import DEFAULT, node, plan, split_recency, validate
from experiments.joint_full_swa_eviction_selection.render import JS


class JointFullSwaTests(unittest.TestCase):
    def test_swa_strip_only_not_full_collapse(self):
        a=plan(DEFAULT);b=plan(DEFAULT,True)
        self.assertEqual(a["freed_units"],8);self.assertEqual(a["retained_full_units"],80)
        self.assertEqual(b["retained_full_units"],48)
        self.assertEqual(a["trace"][0]["component"],"swa")
        self.assertTrue(a["trace"][0]["full_retained"]);self.assertIsNone(a["trace"][0]["swa_tick_after"])

    def test_parent_reentry_and_exhaustion(self):
        p={**DEFAULT,"target_units":1000};a=plan(p)
        self.assertEqual([(x["component"],x["nid"]) for x in a["trace"]],
                         [("swa",1),("swa",3),("full",2),("full",3),("full",1),("full",0)])
        self.assertFalse(a["target_met"]);self.assertEqual(a["freed_units"],96)
        b=plan(p,linear_reference=True)
        self.assertEqual(a["trace"],b["trace"]);self.assertEqual(a["states"],b["states"])

    def test_tie_components_and_swa_lru_rank(self):
        p={**DEFAULT,"nodes":[node(5,None,2,16,2,8,1),node(6,None,3,16,2,8,0)]}
        self.assertEqual(plan(p)["trace"][0]["component"],"full")
        p["nodes"][0]["full_locked"]=True
        self.assertEqual(plan(p)["trace"][0]["nid"],6)

    def test_full_pin_allows_independent_swa_strip(self):
        p={**DEFAULT,"nodes":[{**r,"full_locked":True} if r["nid"]==1 else r for r in DEFAULT["nodes"]]}
        a=plan(p);self.assertEqual(a["trace"][0]["nid"],1);self.assertTrue(a["trace"][0]["full_retained"])
        p["nodes"][1]["swa_locked"]=True
        self.assertEqual(plan(p)["trace"][0]["nid"],3)

    def test_full_cascade_skips_stale_swa_tick(self):
        p={**DEFAULT,"target_units":1000,"nodes":[{**r,"full_tick":0,"swa_tick":20} if r["nid"]==1 else r for r in DEFAULT["nodes"]]}
        a=plan(p);self.assertEqual(a["trace"][0]["component"],"full")
        self.assertGreater(a["stale_entries_skipped"],0)
        self.assertTrue(all(r["swa_tick"] is None for r in a["states"]))

    def test_activation_gates(self):
        for k,v in (("windows",2),("state_slot",True),("chain_allocator",False),("chain_window",False),("chain_mamba",True),("tree_enabled",False),("policy","lru")):
            self.assertEqual(plan({**DEFAULT,k:v})["mode"],"component_lru")
        self.assertEqual(plan({**DEFAULT,"core_supports_joint":False})["mode"],"reject_unsupported_joint")
        self.assertEqual(plan({**DEFAULT,"session_cache":True})["mode"],"reject_unsupported_joint")

    def test_split_recency_and_invalid_forest(self):
        self.assertEqual(split_recency(5,3,20),{"suffix_full_tick":5,"prefix_swa_tick":3,"suffix_swa_tick":3})
        self.assertEqual(split_recency(5,3,20,False)["suffix_full_tick"],20)
        for rows in ([node(1,1,1,1)],[node(1,2,1,1)],[node(1,None,1,1),node(1,None,2,1)]):
            with self.assertRaises(ValueError):validate({**DEFAULT,"nodes":rows})
        with self.assertRaises(ValueError):validate({**DEFAULT,"nodes":[{**DEFAULT["nodes"][1],"swa_tick":None}]})
        with self.assertRaises(ValueError):validate({**DEFAULT,"target_units":-1})

    def test_random_reference_and_node_parity(self):
        rng=random.Random(163);cases=[copy.deepcopy(DEFAULT)]
        for _ in range(64):
            rows=[node(i,None,rng.randrange(16),rng.randrange(1,33),rng.randrange(16) if rng.choice([False,True]) else None,rng.randrange(1,17),i)
                  for i in range(rng.randrange(1,12))]
            for r in rows:r["full_locked"]=rng.choice([False,True]);r["swa_locked"]=rng.choice([False,True])
            p={**DEFAULT,"nodes":rows,"target_units":rng.randrange(256)};cases.append(p)
            a=plan(p);b=plan(p,linear_reference=True)
            self.assertEqual(a["trace"],b["trace"]);self.assertEqual(a["states"],b["states"])
        script=JS+'const fs=require("fs");const ps=JSON.parse(fs.readFileSync(0,"utf8"));console.log(JSON.stringify(ps.map(p=>[plan(p),plan(p,true)])));'
        result=subprocess.run(["node","-e",script],input=json.dumps(cases),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout),[[plan(p),plan(p,True)] for p in cases])


if __name__=="__main__":unittest.main()
