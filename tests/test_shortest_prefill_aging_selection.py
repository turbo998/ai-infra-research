import unittest
import json
import random
import shutil
import subprocess
from experiments.shortest_prefill_aging_selection import run as m
from experiments.shortest_prefill_aging_selection.render import JS


class PrefillAgingTests(unittest.TestCase):
    def test_decode_stale_extend_count_does_not_age(self):
        self.assertEqual(m.plan(m.DEFAULT)["processed_prefill_clock"],1000)
        self.assertEqual(m.plan(m.DEFAULT,legacy_clock=True)["processed_prefill_clock"],3000)
        self.assertEqual(m.plan(m.DEFAULT)["ordered_rids"],["new-short","old-long"])
        self.assertEqual(m.plan(m.DEFAULT,legacy_clock=True)["ordered_rids"],["new-short","old-long"])
        # A common clock shift cancels once both requests have arrived.
        fresh={**m.DEFAULT["requests"][1],"arrival":3000}
        p={**m.DEFAULT,"requests":[m.DEFAULT["requests"][0],fresh],
           "events":[{"is_extend":True,"tokens":3000}]}
        self.assertEqual(m.plan(p)["ordered_rids"],["old-long","new-short"])

    def test_cached_prefix_max_and_minimum_work(self):
        r={**m.DEFAULT["requests"][0],"prompt":256,"output":64,"matched":32,"staged":160}
        self.assertEqual(m.work(r),160)
        self.assertEqual(m.work({**r,"staged":1024}),1)

    def test_deferred_identity_precedes_negative_rank(self):
        old={**m.DEFAULT["requests"][0],"deferred":True}
        p={**m.DEFAULT,"requests":[old,m.DEFAULT["requests"][1]],
           "events":[{"is_extend":True,"tokens":10000}]}
        self.assertEqual(m.plan(p)["ordered_rids"],["new-short","old-long"])

    def test_page_rounding_and_continuation_progress(self):
        r={**m.DEFAULT["requests"][0],"prompt":257}
        p={**m.DEFAULT,"events":[],"factor_numerator":0,"requests":[r]}
        out=m.plan(p)
        self.assertEqual((out["reserved_tokens"],out["continuation_limit"]),(512,3584))
        self.assertIsNone(m.plan({**p,"budget":511})["continuation_limit"])

    def test_explicit_invalid_factors_and_future_affine_guard(self):
        for change in ({"factor_numerator":-1},{"factor_numerator":True},{"factor_denominator":0},
                       {"policy":"fcfs"},{"page":0},{"events":[{"is_extend":1,"tokens":1}]}):
            with self.assertRaises(ValueError):m.plan({**m.DEFAULT,**change})
        with self.assertRaises(ValueError):
            m.affine_order(m.DEFAULT["requests"],0,1,2)

    def test_no_age_preserves_work_order_and_budget_stop(self):
        p={**m.DEFAULT,"factor_numerator":0,"events":[{"is_extend":True,"tokens":10000}]}
        self.assertEqual(m.plan(p)["ordered_rids"],["new-short","old-long"])
        huge={**m.DEFAULT["requests"][0],"prompt":5000}
        self.assertIsNone(m.plan({**p,"requests":[huge]})["continuation_limit"])

    def test_random_all_arrived_clock_cancellation(self):
        rng=random.Random(161)
        for _ in range(64):
            rows=[{**m.DEFAULT["requests"][0],"rid":str(i),"prompt":rng.randrange(1,2048),
                   "arrival":rng.randrange(2049),"entry":i,"deferred":bool(rng.randrange(2))} for i in range(8)]
            full=sorted(rows,key=lambda r:(r["deferred"],m.rank_numerator(r,2048,1,2),r["entry"]))
            self.assertEqual(m.affine_order(rows,2048,1,2),full)
            self.assertEqual(m.affine_order(rows,8192,1,2),full)

    @unittest.skipUnless(shutil.which("node"),"Node required for original H5parity")
    def test_node_rational_clock_page_parity(self):
        rng=random.Random(1611);cases=[m.DEFAULT]
        for _ in range(64):
            p={**m.DEFAULT,"factor_numerator":rng.randrange(3),"budget":rng.randrange(1,33)*256,
               "events":[{"is_extend":True,"tokens":rng.randrange(4097)}]}
            cases.append(p)
        script=JS+"\nconst p=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(p.map(v=>[plan(v),plan(v,true),plan(v,false,true)])));"
        result=subprocess.run(["node","-e",script],input=json.dumps(cases),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout),[[m.plan(p),m.plan(p,True),m.plan(p,without_aging=True)] for p in cases])


if __name__=="__main__":unittest.main()
