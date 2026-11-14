import bisect
import copy
import json
import random
import subprocess
import unittest
from experiments.managed_engine_drain_selection.run import DEFAULT,plan
from experiments.managed_engine_drain_selection.render import JS


class ManagedDrainTests(unittest.TestCase):
    def test_frontend_drain_then_launcher_not_group(self):
        a=plan(DEFAULT);b=plan(DEFAULT,"legacy_group_first")
        self.assertEqual((a["completed_requests"],b["completed_requests"]),(2,0))
        self.assertEqual(a["engine_start_ms"],6000)
        self.assertIn({"at_ms":6000,"event":"TERM_launcher"},a["events"])
        self.assertEqual((a["stragglers_remaining"],b["stragglers_remaining"]),(0,1))

    def test_minimum_can_exceed_remaining_deadline(self):
        a=plan({**DEFAULT,"launcher_exit_after_ms":None})
        self.assertEqual((a["remaining_budget_ms"],a["launcher_wait_ms"]),(4000,5000))
        self.assertEqual((a["modeled_end_ms"],a["modeled_deadline_overrun_ms"]),(11000,1000))
        self.assertIn("KILL_group_timeout",[x["event"] for x in a["events"]])
        self.assertIn("KILL_group_post_exit",[x["event"] for x in a["events"]])

    def test_zero_budget_and_headless(self):
        self.assertEqual(plan({**DEFAULT,"budget_ms":0,"launcher_exit_after_ms":None})["launcher_wait_ms"],5000)
        p={**DEFAULT,"has_api":False,"api_drain_ms":0,"launcher_exit_after_ms":7000}
        a=plan(p)
        self.assertEqual((a["engine_start_ms"],a["completed_requests"]),(0,2))

    def test_serve_failure_preserved_after_cleanup(self):
        a=plan({**DEFAULT,"serve_error":True})
        self.assertEqual(a["outcome"],"serve_error_after_cleanup")
        self.assertEqual(a["events"][-1]["event"],"preserve_serve_error")
        b=plan({**DEFAULT,"reason":"server"})
        self.assertEqual(b["engine_start_ms"],0)
        self.assertEqual(b["outcome"],"serve_error_after_cleanup")

    def test_noop_keeps_completion_unknown(self):
        for key in ("already_started","pid_present"):
            p={**DEFAULT,key:key=="already_started"}
            a=plan(p)
            self.assertEqual(a["events"],[])
            self.assertIsNone(a["completed_requests"])
            self.assertIsNone(a["stragglers_remaining"])

    def test_declared_finish_ties_and_sorted_prefix_oracle(self):
        p={**DEFAULT,"has_api":False,"api_drain_ms":0,"launcher_exit_after_ms":5000,"requests":[4999,5000,5001]}
        self.assertEqual(plan(p)["completed_requests"],2)
        rng=random.Random(168)
        for _ in range(64):
            xs=sorted(rng.randrange(10001) for _ in range(rng.randrange(1,65)))
            cutoff=rng.randrange(10001)
            self.assertEqual(bisect.bisect_right(xs,cutoff),sum(x<=cutoff for x in xs))

    def test_invalid_lifecycle_inputs_fail_explicitly(self):
        for change in ({"budget_ms":True},{"requests":[]},{"launcher_exit_after_ms":-1},
                       {"has_api":False},{"reason":"unknown"},{"reason":"engine_exit"},{"stragglers":17}):
            with self.subTest(change=change),self.assertRaises(ValueError):plan({**DEFAULT,**change})

    def test_node_parity_on65profiles(self):
        rng=random.Random(168);profiles=[copy.deepcopy(DEFAULT)]
        for _ in range(64):
            has_api=rng.choice([True,False])
            reason=rng.choice(["signal","server","engine_exit"])
            profiles.append({**DEFAULT,"has_api":has_api,"api_drain_ms":rng.randrange(15001) if has_api else 0,
                             "budget_ms":rng.randrange(15001),"launcher_exit_after_ms":rng.choice([None,0,1000,5000,20000]),
                             "requests":[rng.randrange(20001) for _ in range(rng.randrange(1,9))],
                             "stragglers":rng.randrange(4),"pid_present":False if reason=="engine_exit" else rng.choice([True,True,False]),
                             "already_started":rng.choice([False,False,True]),"reason":reason,
                             "serve_error":rng.choice([True,False])})
        expected=[[plan(p,s) for s in ("drain_first","legacy_group_first")] for p in profiles]
        script=JS+"\nconst ps="+json.dumps(profiles)+";console.log(JSON.stringify(ps.map(p=>['drain_first','legacy_group_first'].map(s=>plan(p,s)))));"
        out=subprocess.run(["node"],input=script,check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(out.stdout),expected)
