from itertools import permutations
import json
import statistics
import subprocess
import unittest

from experiments.lora_hrw_sticky_selection import run as m
from experiments.lora_hrw_sticky_selection.render import JS


class LoraHrwTests(unittest.TestCase):
    def test_sticky_subset_oracle_across_capacity_patterns(self):
        prior=[[9,0],[10,0]]
        for mask in range(16):
            workers=[{**w,"used":2 if mask&(1<<i) else 0} for i,w in enumerate(m.WORKERS)]
            for replicas in range(6):
                for sticky in (False,True):
                    self.assertEqual(m.allocate(workers,replicas,prior,sticky)["selected"],m.independent_subset_oracle(workers,replicas,prior,sticky))

    def test_forced_tie_total_order_and_pair_identity(self):
        workers=[{**w,"score":100} for w in m.WORKERS]
        expected=[[7,0],[7,1],[9,0],[10,0]]
        for order in permutations(workers):
            self.assertEqual(m.allocate(list(order),4,sticky=False)["selected"],expected)

    def test_eligible_prior_retention_and_full_replacement(self):
        prior=[[9,0],[10,0]]
        self.assertEqual(m.allocate(m.WORKERS,2,prior)["selected"],prior)
        workers=[{**w,"used":2 if w["id"]==9 else 0} for w in m.WORKERS]
        self.assertEqual(m.allocate(workers,2,prior)["selected"],[[10,0],[7,0]])

    def test_partial_underfill_all_full_tag_and_unknown(self):
        one=[{**w,"used":0 if w["id"]==10 else 2} for w in m.WORKERS]
        self.assertEqual(m.allocate(one,2)["selected"],[[10,0]])
        allfull=[{**w,"used":2} for w in m.WORKERS]
        out=m.allocate(allfull,2)
        self.assertEqual(out["fallback"],"all_workers_full_basic_hrw")
        self.assertEqual(out["selected_full_count"],2)
        unknown=[{**w,"used":None,"capacity":None} for w in m.WORKERS]
        self.assertEqual(m.allocate(unknown,2)["unknown_slot_workers"],4)
        self.assertIsNone(m.allocate(unknown,2)["fallback"])

    def test_membership_changes_and_empty_zero(self):
        original=m.allocate(m.WORKERS,2,sticky=False)["selected"]
        added=m.WORKERS+[{"id":11,"dp":0,"score":110,"used":0,"capacity":2}]
        after=m.allocate(added,2,sticky=False)["selected"]
        self.assertIn([11,0],after)
        self.assertEqual(len([w for w in original if w not in after]),1)
        self.assertEqual(m.allocate(m.WORKERS[:-1],2,sticky=False)["selected"],original)
        self.assertEqual(m.allocate([],2)["selected"],[])
        self.assertEqual(m.allocate(m.WORKERS,0)["selected"],[])

    def test_invalid_identity_slots_and_budget(self):
        for workers in (m.WORKERS+[m.WORKERS[0]],[{**m.WORKERS[0],"capacity":None}],[{**m.WORKERS[0],"score":2**53}]):
            with self.assertRaises(ValueError):
                m.allocate(workers,2)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"replicas":2**53})

    def test_node_placement_and_budget_parity(self):
        for workers in (m.WORKERS,[{**w,"used":2} for w in m.WORKERS],[{**w,"used":None,"capacity":None} for w in m.WORKERS]):
            source=JS+"\nconsole.log(JSON.stringify({result:simulate("+json.dumps(workers)+",2,[[9,0],[10,0]],true),budget:budget("+json.dumps(m.DEFAULT)+")}))"
            out=json.loads(subprocess.check_output(["node","-e",source],text=True))
            self.assertEqual(out["result"],m.allocate(workers,2,[[9,0],[10,0]]))
            self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_timing_paths(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("topk","sort"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(128);self.assertEqual(a(),b())
        self.assertIsNone(report["hash_distribution_quality"])


if __name__=="__main__":
    unittest.main()
