import copy
import json
import random
import subprocess
import unittest

from experiments.ragged_alltoall_contract_selection import run as m
from experiments.ragged_alltoall_contract_selection.render import JS


class RaggedAlltoallTests(unittest.TestCase):
    def test_independent_direct_owner_oracle(self):
        rng=random.Random(144)
        for _ in range(24):
            world=rng.randint(1,6);tokens=[[rng.randint(-9,9) for _ in range(rng.randrange(9))] for _ in range(world)]
            dest=[[rng.randrange(world) for _ in row] for row in tokens]
            result=m.dispatch({"tokens":tokens,"destinations":dest})
            expected=[[2*v+100*d for v,d in zip(values,targets)] for values,targets in zip(tokens,dest)]
            self.assertEqual(result["restored"],expected)
            for d,records in enumerate(result["received"]):
                expected_ids=[(source,i) for source in range(world) for i in range(len(tokens[source])) if dest[source][i]==d]
                self.assertEqual([(r["owner"],r["position"]) for r in records],expected_ids)

    def test_transpose_contract_and_same_total_mismatch(self):
        r=m.dispatch(m.DEFAULT)
        self.assertEqual(r["recv_splits"],[[r["send_splits"][s][d] for s in range(4)] for d in range(4)])
        bad=copy.deepcopy(r["recv_splits"]);bad[0][0],bad[0][2]=bad[0][2],bad[0][0]
        self.assertEqual(sum(bad[0]),sum(r["recv_splits"][0]))
        with self.assertRaisesRegex(ValueError,"pairwise split mismatch"):
            m.dispatch(m.DEFAULT,bad)

    def test_zero_receive_rank_can_still_send(self):
        r=m.dispatch(m.DEFAULT)
        self.assertEqual(r["receive_rows"][1],0);self.assertEqual(r["send_rows"][1],2)
        self.assertEqual(r["received"][1],[]);self.assertEqual(len(r["restored"][1]),2)
        empty=m.dispatch({"tokens":[[],[]],"destinations":[[],[]]})
        self.assertEqual(empty["tokens"],0);self.assertEqual(empty["restored"],[[],[]])

    def test_inverse_identity_not_payload_or_arrival_order(self):
        r=m.dispatch(m.DEFAULT)
        self.assertEqual(r["naive_positions_matching"],4)
        self.assertEqual(m.restore([list(reversed(v)) for v in r["responses"]],[3,2,1,2]),r["restored"])
        duplicate=m.dispatch({"tokens":[[5,5],[5]],"destinations":[[1,0],[0]]})
        self.assertEqual(duplicate["restored"],[[110,10],[10]])

    def test_missing_duplicate_and_bad_group_rank_are_explicit(self):
        r=m.dispatch(m.DEFAULT);responses=copy.deepcopy(r["responses"]);responses[0].append(copy.deepcopy(responses[0][0]))
        with self.assertRaises(ValueError):
            m.restore(responses,[3,2,1,2])
        responses=copy.deepcopy(r["responses"]);responses[0].pop()
        with self.assertRaises(ValueError):
            m.restore(responses,[3,2,1,2])
        for p in ({"tokens":[[1]],"destinations":[[1]]},{"tokens":[[True]],"destinations":[[0]]},{"tokens":[[1]],"destinations":[[]]}):
            with self.assertRaises(ValueError):
                m.dispatch(p)

    def test_selected_payload_units(self):
        r=m.budget({"world":4,"tokens":8,"features":4,"element_bytes":2})
        self.assertEqual(r["selected_one_way_token_payload_bytes"],64)
        self.assertEqual(r["selected_roundtrip_token_payload_bytes"],128)
        self.assertEqual(r["selected_all_rank_send_and_recv_int64_split_bytes"],256)
        self.assertEqual(r["selected_one_way_two_int64_identity_bytes"],128)
        with self.assertRaises(ValueError):
            m.budget({"world":True,"tokens":8,"features":4,"element_bytes":2})

    def test_node_python_dispatch_parity(self):
        for p in (m.DEFAULT,{"tokens":[[],[]],"destinations":[[],[]]}):
            result=subprocess.run(["node","-e",JS+"\nconsole.log(JSON.stringify(dispatch("+json.dumps(p)+")));"],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout),m.dispatch(p))
        bad=copy.deepcopy(m.dispatch(m.DEFAULT)["recv_splits"]);bad[0][0],bad[0][2]=bad[0][2],bad[0][0]
        result=subprocess.run(["node","-e",JS+"\ndispatch("+json.dumps(m.DEFAULT)+","+json.dumps(bad)+");"],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn("Pairwise split mismatch",result.stderr)

    def test_sealed_report_and_cpu_paths(self):
        for n in (64,256,1024,4096):
            a,b=m.timing_paths(n);self.assertEqual(a(),b())
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        for row in latency["rows"]:
            for path in ("indexed","searching"):
                samples=row[path+"_samples_ms"];self.assertEqual(len(samples),7)
                self.assertEqual(sorted(samples)[3],row[path+"_median_ms"])
        self.assertEqual(json.loads((m.HERE/"report.json").read_text(encoding="utf-8")),m.build_report(latency))
