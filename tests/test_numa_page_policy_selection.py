import copy
import json
import random
import subprocess
import unittest

from experiments.numa_page_policy_selection import run as m
from experiments.numa_page_policy_selection.render import JS


class NumaPagePolicyTests(unittest.TestCase):
    def test_independent_expanded_schedule_oracle(self):
        rng=random.Random(141)
        for _ in range(30):
            nodes=sorted(rng.sample(range(32),rng.randint(1,8)))
            weights=[rng.randint(1,9) for _ in nodes]
            expanded=[node for node,w in zip(nodes,weights) for _ in range(w)]
            for offset in range(128):
                self.assertEqual(m.weighted_node(offset,nodes,weights),expanded[offset%len(expanded)])

    def test_default_locality_is_workload_dependent(self):
        out=m.simulate(m.DEFAULT)
        self.assertEqual([out[k]["remote_reads"] for k in out],[9,0,5,4])
        self.assertEqual(out["weighted"]["node_pages"],{"0":3,"1":9})
        p=copy.deepcopy(m.DEFAULT);p["reader_nodes"]=[0]*12
        changed=m.simulate(p)
        self.assertEqual(changed["serial_first_write"]["remote_reads"],0)
        self.assertEqual(changed["weighted"]["remote_reads"],9)

    def test_weighted_fault_order_is_page_indexed(self):
        forward=m.Pages("weighted",[2,5],[1,3]);reverse=m.Pages("weighted",[2,5],[1,3])
        for i in range(20):
            forward.first_write(i,2)
        for i in reversed(range(20)):
            reverse.first_write(i,5)
        self.assertEqual(forward.owners,reverse.owners)
        self.assertEqual([forward.owners[i] for i in range(4)],[2,5,5,5])

    def test_policy_change_does_not_migrate_or_duplicate(self):
        pages=m.Pages("local",[0,1],[1,1])
        for i in range(6):
            pages.first_write(i,0)
        pages.set_policy("weighted",[0,1],[1,3])
        for i in reversed(range(12)):
            pages.first_write(i,1)
        self.assertEqual([pages.owners[i] for i in range(6)],[0]*6)
        self.assertEqual([pages.owners[i] for i in range(6,12)],[1,1,0,1,1,1])
        self.assertEqual(len(pages.owners),12)

    def test_effective_mask_and_payload_conservation(self):
        nodes,weights=m.effective_nodes([2,5],[1,3],[5])
        self.assertEqual([m.weighted_node(i,nodes,weights) for i in range(8)],[5]*8)
        with self.assertRaises(ValueError):
            m.effective_nodes([2,5],[1,3],[9])
        for result in m.simulate(m.DEFAULT).values():
            self.assertEqual(sum(result["node_pages"].values()),12)
            self.assertEqual(sum(result["node_payload_bytes"].values()),49152)
            self.assertEqual(result["local_reads"]+result["remote_reads"],12)

    def test_invalid_inputs_are_explicit(self):
        for nodes,weights in (([5,2],[1,3]),([2,2],[1,3]),([2],[0]),([True],[1]),([2],[256]),([],[])):
            with self.assertRaises(ValueError):
                m.weighted_node(0,nodes,weights)
        for key,value in (("pages",0),("pages",True),("reader_nodes",[0]),("page_bytes",1000),("initial_writer",-1)):
            p=copy.deepcopy(m.DEFAULT);p[key]=value
            with self.assertRaises(ValueError):
                m.simulate(p)
        with self.assertRaises(ValueError):
            m.weighted_node(-1,[0],[1])

    def test_node_python_placement_parity_and_bad_input(self):
        p=copy.deepcopy(m.DEFAULT);p["nodes"]=[2,5];p["weights"]=[1,3]
        script=JS+"\nconsole.log(JSON.stringify(simulate("+json.dumps(p)+")));"
        result=subprocess.run(["node","-e",script],text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout),m.simulate(p))
        bad=copy.deepcopy(p);bad["weights"]=[0,3]
        result=subprocess.run(["node","-e",JS+"\nsimulate("+json.dumps(bad)+");"],text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("weight",result.stderr)

    def test_sealed_report_and_timing_paths(self):
        for n in (64,256,1024,4096):
            a,b=m.timing_paths(n);self.assertEqual(a(),b())
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        for row in latency["rows"]:
            for path in ("prefix","walk"):
                samples=row[path+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(sorted(samples)[3],row[path+"_median_ms"])
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        self.assertIsNone(report["native_memory_bandwidth_gbps"])
