import copy
import json
import random
import statistics
import subprocess
import unittest

from experiments.nccl_shrink_ownership_selection import run as m
from experiments.nccl_shrink_ownership_selection.render import JS,cards


class ShrinkOwnershipTests(unittest.TestCase):
    def test_independent_rank_compression_and_cyclic_slice_oracle(self):
        rng=random.Random(146)
        for _ in range(40):
            world=rng.randrange(2,9);excluded=rng.sample(range(world),rng.randrange(1,world))
            globals_=rng.sample(range(100,1000),world);values=[rng.randrange(5) for _ in range(rng.randrange(1,33))]
            survivors=[rank for rank in range(world) if rank not in excluded]
            p={**m.DEFAULT,"parent_global_ranks":globals_,"excluded":excluded,"callers":list(reversed(survivors)),
               "rows":values,"checkpoint":[[i,v] for i,v in enumerate(values)]}
            result=m.simulate(p)
            expected=[{"parent_rank":old,"new_rank":new,"global_id":globals_[old]} for new,old in enumerate(survivors)]
            self.assertEqual(result["membership"]["members"],expected)
            slices=[[[i,values[i]] for i in range(rank,len(values),len(survivors))] for rank in range(len(survivors))]
            self.assertEqual(result["resharded"],slices)
            self.assertEqual(sorted(r[0] for rows in result["resharded"] for r in rows),list(range(len(values))))

    def test_default_retained_and_lost_rows_are_not_automatic_recovery(self):
        result=m.simulate(m.DEFAULT);budget=result["selected_budget"]
        self.assertEqual(result["lost_row_ids"],[1,4,7,10])
        self.assertEqual((budget["already_at_target_rows"],budget["retained_rows"],budget["survivor_rows_needing_move"]),(2,8,6))
        missing=m.simulate({**m.DEFAULT,"checkpoint":[]})
        self.assertFalse(missing["data_ready"]);self.assertIsNone(missing["resharded"])
        self.assertIn("[1,4,7,10]",missing["data_error"])

    def test_excluded_and_missing_callers_are_explicit_errors(self):
        for callers in ([0,1,2,3,5],[0,2,3],[0,2,3,5,5]):
            with self.assertRaises(ValueError):
                m.simulate({**m.DEFAULT,"callers":callers})
        for excluded in ([],list(range(6)),[1,1],[True],[6]):
            with self.assertRaises(ValueError):
                m.simulate({**m.DEFAULT,"excluded":excluded})

    def test_default_abort_and_share_request_scope(self):
        with self.assertRaisesRegex(ValueError,"outstanding"):
            m.simulate({**m.DEFAULT,"outstanding":2})
        result=m.simulate({**m.DEFAULT,"outstanding":2,"mode":"abort"})
        self.assertTrue(result["membership"]["abort_requested"])
        self.assertFalse(result["membership"]["selected_sharing_request_permitted"])
        self.assertTrue(m.simulate(m.DEFAULT)["membership"]["selected_sharing_request_permitted"])
        with self.assertRaises(ValueError):
            m.simulate({**m.DEFAULT,"shrink_share":1})

    def test_exclusion_order_global_labels_checkpoint_integrity(self):
        result=m.simulate(m.DEFAULT)
        self.assertEqual(result,m.simulate({**m.DEFAULT,"excluded":[4,1],"callers":[5,3,2,0]}))
        self.assertEqual([v["global_id"] for v in result["membership"]["members"]],[7,19,5,17])
        for records in ([[1,0]],[[1,101],[1,101]],[[12,0]]):
            with self.assertRaises(ValueError):
                m.simulate({**m.DEFAULT,"checkpoint":records})
        empty=m.simulate({**m.DEFAULT,"rows":[],"checkpoint":[]})
        self.assertTrue(empty["data_ready"]);self.assertEqual(empty["resharded"],[[],[],[],[]])

    def test_selected_payload_and_mapping_paths(self):
        budget=m.simulate(m.DEFAULT)["selected_budget"]
        self.assertEqual((budget["selected_total_feature_bytes"],budget["selected_survivor_move_feature_bytes"],
                          budget["selected_checkpoint_restore_feature_bytes"]),(72,36,24))
        for n in (4,17,64,256):
            indexed,counted=m.timing_paths(n);self.assertEqual(indexed(),counted())

    def test_node_python_membership_data_gate_parity(self):
        plans=[v["inputs"] for v in m.cases() if not v["preflight_error"]]
        rng=random.Random(146)
        for _ in range(20):
            p=copy.deepcopy(m.DEFAULT);rng.shuffle(p["excluded"]);rng.shuffle(p["callers"])
            p["rows"]=[rng.randrange(3) for _ in range(rng.randrange(0,33))]
            p["checkpoint"]=[[i,v] for i,v in enumerate(p["rows"])] if rng.randrange(2) else []
            plans.append(p)
        result=subprocess.run(["node"],input=JS+"\nconsole.log(JSON.stringify("+json.dumps(plans)+".map(simulate)));",capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[m.simulate(p) for p in plans])
        p={**m.DEFAULT,"callers":[0,1,2,3,5]}
        result=subprocess.run(["node"],input=JS+"\nsimulate("+json.dumps(p)+");",capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn("excluded ranks must not call",result.stderr)

    def test_sealed_original_cpu_and_full_matrix(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency));self.assertIsNone(report["native_shrink_latency_ms"])
        self.assertEqual(cards(report).split('<section class="card">')[3].count("<tr>"),len(report["scenarios"])+1)
        for row in latency["rows"]:
            for name in ("prefix","counted"):
                samples=row[name+"_samples_ms"]
                self.assertEqual(len(samples),7);self.assertTrue(all(v>0 for v in samples))
                self.assertEqual(row[name+"_median_ms"],statistics.median(samples))


if __name__=="__main__":
    unittest.main()
