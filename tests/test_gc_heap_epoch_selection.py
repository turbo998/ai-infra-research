import json
import random
import subprocess
import sys
import unittest
from experiments.gc_heap_epoch_selection.run import DEFAULT,plan,validate
from experiments.gc_heap_epoch_selection.render import JS


class GCHeapEpochTests(unittest.TestCase):
    def test_startup_skip_visits_and_warmup_reset(self):
        a=plan(DEFAULT);b=plan(DEFAULT,True);s=plan(DEFAULT,skip_epoch_reset=True)
        self.assertEqual((a["modeled_init_object_visits"],b["modeled_init_object_visits"]),(0,192))
        self.assertEqual(a["final_frozen_nodes"],80);self.assertEqual(a["collected_cycle_nodes"],24)
        self.assertEqual(s["unreachable_frozen_cycles"],16)

    def test_capture_default_exits_startup_epoch(self):
        a=plan({**DEFAULT,"captures":1});tr=a["trace"]
        self.assertFalse(tr[2]["automatic_gc_enabled"]);self.assertTrue(tr[3]["automatic_gc_enabled"])
        self.assertEqual(tr[3]["frozen_nodes"],0);self.assertEqual(tr[3]["unreachable_frozen_cycles"],0)

    def test_optout_preserves_startup_until_warmup(self):
        a=plan({**DEFAULT,"captures":1,"capture_gc_opt_out":True})
        self.assertEqual(a["trace"][3]["frozen_nodes"],64)
        self.assertEqual(a["trace"][3]["unreachable_frozen_cycles"],16)
        self.assertEqual(a["unreachable_frozen_cycles"],0)

    def test_prior_disabled_state_and_zero_counts(self):
        a=plan({**DEFAULT,"captures":2,"gc_initially_enabled":False})
        self.assertTrue(all(not x["automatic_gc_enabled"] for x in a["trace"]))
        z={**DEFAULT,"startup_live":0,"drop_startup_cycles":0,"late_live":0,"late_garbage":0}
        self.assertEqual(plan(z)["final_frozen_nodes"],0)

    def test_invalid_epoch_contract(self):
        for k,v in (("drop_startup_cycles",65),("startup_live",True),("captures",9),("capture_gc_opt_out",1),("late_garbage",-1)):
            with self.assertRaises(ValueError):validate({**DEFAULT,k:v})
        with self.assertRaises(ValueError):validate({**DEFAULT,"unexpected":1})

    def test_isolated_cpython_cycle_refcount_probe(self):
        p=subprocess.run([sys.executable,"-m","experiments.gc_heap_epoch_selection.run","--probe"],text=True,capture_output=True,check=True)
        a=json.loads(p.stdout)
        self.assertEqual(a["frozen_cycles_alive_after_collection"],64)
        self.assertEqual(a["cycles_alive_after_unfreeze_collection"],0)
        self.assertTrue(a["frozen_acyclic_refcount_freed"]);self.assertTrue(a["postfreeze_new_cycle_collected"])
        self.assertGreater(a["actual_total_freeze_count"],64)

    def test_isolated_finalizer_gen0_cleanup(self):
        p=subprocess.run([sys.executable,"-m","experiments.gc_heap_epoch_selection.run","--finalizer-probe"],text=True,capture_output=True,check=True)
        a=json.loads(p.stdout)["results"]
        self.assertEqual(a["old_0_1_2"]["finalizer_created_cycles"],1)
        self.assertEqual(a["old_0_1_2"]["alive_when_frozen"],1)
        self.assertEqual(a["full_then_0"]["alive_when_frozen"],0)

    def test_node_65configs_195outcomes(self):
        rng=random.Random(164);cases=[DEFAULT]
        for _ in range(64):
            n=rng.randrange(256)
            cases.append({**DEFAULT,"startup_live":n,"drop_startup_cycles":rng.randrange(n+1),
                          "late_live":rng.randrange(128),"late_garbage":rng.randrange(32),"captures":rng.randrange(4),
                          "init_collections":rng.randrange(8),"capture_gc_opt_out":rng.choice([False,True]),
                          "gc_initially_enabled":rng.choice([False,True])})
        script=JS+'const fs=require("fs");const ps=JSON.parse(fs.readFileSync(0,"utf8"));console.log(JSON.stringify(ps.map(p=>[plan(p),plan(p,true),plan(p,false,true)])));'
        a=subprocess.run(["node","-e",script],input=json.dumps(cases),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(a.stdout),[[plan(p),plan(p,True),plan(p,skip_epoch_reset=True)] for p in cases])


if __name__=="__main__":unittest.main()
