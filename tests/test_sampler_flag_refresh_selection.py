import json
import random
import subprocess
import unittest

from experiments.sampler_flag_refresh_selection import run as m
from experiments.sampler_flag_refresh_selection.render import JS,cards


class SamplerRefreshTests(unittest.TestCase):
    def test_safe_sticky_whole_vocab_refresh(self):
        p={**m.DEFAULT,"top_ks":[1,m.ALL],"keep":[1]}
        r=m.evaluate(p)
        self.assertTrue(r["before"]["need_top_k_sampling"])
        self.assertFalse(r["after"]["need_top_k_sampling"])
        self.assertTrue(r["matches_fresh_nonempty_flags"])
        self.assertEqual(r["support_misses"],0)
        self.assertEqual(r["rows"][0]["refreshed"]["route"],"whole_vocab_direct")

    def test_independent_tied_support_and_valid_upper_bound(self):
        scores=[3,7,7,1];f=m.flags([m.ALL,2]);next_f=m.flags([2])
        self.assertEqual(m.support(scores,2,f,False,False)["support_token_ids"],[1,2])
        self.assertEqual(m.support(scores,2,next_f,False,False)["support_token_ids"],[1,2])
        broad={**next_f,"max_top_k":3}
        self.assertEqual(m.support(scores,2,broad,False,False)["support_token_ids"],[1,2])

    def test_greedy_and_empty_scheduler_short_circuit(self):
        r=m.evaluate({**m.DEFAULT,"top_ks":[m.ALL,1],"keep":[1]})
        self.assertTrue(r["after"]["is_all_greedy"])
        self.assertEqual(r["rows"][0]["refreshed"]["route"],"greedy_argmax")
        empty=m.evaluate({**m.DEFAULT,"keep":[]})
        self.assertEqual(empty["status"],"empty_no_forward")
        self.assertEqual(empty["before"],empty["after"])
        self.assertIsNone(empty["fresh_nonempty_flags"])
        self.assertEqual(empty["rows"],[])

    def test_seed_capture_general_path_keeps_full_ordering(self):
        for changes in ({"seeded":True},{"capture":True},{"seeded":True,"capture":True}):
            r=m.evaluate({**m.DEFAULT,**changes})
            self.assertEqual(r["after"]["max_top_k"],20)
            self.assertEqual(r["rows"][0]["refreshed"]["route"],"full_sort")
            self.assertEqual(r["rows"][0]["refreshed"]["planned_ranked_prefix_columns"],2048)
            self.assertEqual(r["support_misses"],0)

    def test_npu_eligibility_not_native_availability(self):
        r=m.evaluate({**m.DEFAULT,"top_ks":[1300,20]})
        self.assertFalse(r["before"]["npu_top_k_top_p_eligible"])
        self.assertTrue(r["after"]["npu_top_k_top_p_eligible"])
        self.assertEqual((r["before"]["max_top_k"],r["after"]["max_top_k"]),(1300,20))
        self.assertIsNone(r["native_npu_execution"])

    def test_invalid_unresolved_rows_keeps_and_unsafe_cap(self):
        for changes in ({"top_ks":[-1]},{"top_ks":[True]},{"keep":[1,1]},{"keep":[2]},
                        {"keep":[1,0]},{"seeded":1},{"vocab":0}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                m.evaluate({**m.DEFAULT,**changes})
        with self.assertRaises(ValueError):
            m.support([1,2,3,4],2,{**m.flags([2]),"max_top_k":1},False,False)

    def test_random_filters_reversed_requests_and_node_parity(self):
        rng=random.Random(154);inputs=[m.DEFAULT,{**m.DEFAULT,"seeded":True},{**m.DEFAULT,"capture":True},{**m.DEFAULT,"keep":[]}]
        for _ in range(32):
            ks=[rng.choice([1,2,3,16,m.ALL]) for _ in range(rng.randrange(1,9))]
            keep=[i for i in range(len(ks)) if rng.randrange(2)]
            inputs.append({**m.DEFAULT,"vocab":64,"top_ks":ks,"keep":keep})
        a=m.evaluate(m.DEFAULT);b=m.evaluate({**m.DEFAULT,"top_ks":[20,m.ALL],"keep":[0]})
        self.assertEqual(a["after"],b["after"])
        self.assertEqual(a["rows"][0]["refreshed"],b["rows"][0]["refreshed"])
        script=JS+"\nconsole.log(JSON.stringify("+json.dumps(inputs)+".map(evaluate)));"
        result=subprocess.run(["node"],input=script,capture_output=True,text=True,check=True)
        expected=[m.evaluate(p) for p in inputs]
        self.assertEqual(json.loads(result.stdout),expected)
        for r in expected:
            self.assertEqual(r["support_misses"],0)
            if r["retained_rows"]:
                self.assertTrue(r["matches_fresh_nonempty_flags"])

    def test_sealed_matrix_and_cpu_paths(self):
        latency=json.loads((m.HERE/"latency.json").read_text());sealed=json.loads((m.HERE/"report.json").read_text())
        self.assertEqual(sealed,m.build_report(latency))
        self.assertEqual(sealed["summary"],{"retained_rows":6,"support_misses":0,"sticky_prefix_columns":9542,
                                           "refreshed_prefix_columns":1114,"tightened_plans":5})
        for n in (64,256,1024,4096):
            self.assertEqual(m.bounded_prefix(m.weights(n),20),m.full_prefix(m.weights(n),20))
        self.assertIsNone(sealed["native_seeded_tie_order"])
        text=cards(sealed);self.assertEqual(text.count('class="card"'),7)
        self.assertNotIn("<script",text)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",text)
