import copy
import itertools
import json
import random
import statistics
import subprocess
import unittest

from experiments.nccl2_process_key_gate_selection import run as m
from experiments.nccl2_process_key_gate_selection.render import JS,cards


class ProcessKeyGateTests(unittest.TestCase):
    def test_independent_four_state_lifecycle_oracle(self):
        # Declarative oracle has no byte/version/backend implementation.
        table={
            "V":{"unset":("L",None),"A":("A",None),"B":("B",None),"envA":("A",None),"envB":("B",None),"destroy":("V",None),"failA":("V","injected_native_api_failure")},
            "L":{"unset":("L",None),"A":("L","first_nccl2_group_required"),"B":("L","first_nccl2_group_required"),"envA":("L","first_nccl2_group_required"),"envB":("L","first_nccl2_group_required"),"destroy":("L",None),"failA":("L","first_nccl2_group_required")},
            "A":{"unset":("A",None),"A":("A",None),"B":("A","different_process_key"),"envA":("A",None),"envB":("A",None),"destroy":("A",None),"failA":("A",None)},
            "B":{"unset":("B",None),"A":("B","different_process_key"),"B":("B",None),"envA":("B",None),"envB":("B",None),"destroy":("B",None),"failA":("B","different_process_key")},
        }
        events={"unset":m.event(),"A":m.event(m.key()),"B":m.event(m.key("synthetic-B")),
                "envA":m.event(None,m.key()),"envB":m.event(None,m.key("synthetic-B")),
                "destroy":m.event(operation="destroy"),"failA":m.event(m.key(),api_ok=False)}
        for size in range(5):
            for names in itertools.product(events,repeat=size):
                p={**m.DEFAULT,"events":[events[name] for name in names]};actual=m.simulate(p);state="V";errors=[]
                for name in names:
                    state,error=table[state][name];errors.append(error)
                self.assertEqual([s["error"] for s in actual["steps"]],errors,names)
                self.assertEqual(actual["final_state"]["created_nccl2"],state!="V",names)
                applied=actual["final_state"]["applied"]
                self.assertEqual(None if applied is None else applied["cohort"],"synthetic-"+state if state in ("A","B") else None,names)

    def test_minimum_length_nul_and_failure_rollback(self):
        for size in (0,1,31,32,64,256):
            p={**m.DEFAULT,"events":[m.event(m.key(byte_length=size)),m.event(m.key())]}
            result=m.simulate(p)
            self.assertEqual(result["steps"][0]["error"],"injected_native_api_failure" if size<32 else None)
            self.assertIsNotNone(result["final_state"]["applied"])
        p={**m.DEFAULT,"events":[m.event(m.key(nul=True)),m.event(m.key(),api_ok=False),m.event(m.key())]}
        result=m.simulate(p)
        self.assertEqual(result["rejections"],2);self.assertEqual(result["final_state"]["applied"],m.key())

    def test_option_precedence_and_no_environment_reread(self):
        p={**m.DEFAULT,"events":[m.event(m.key(),m.key("synthetic-B")),m.event(None,m.key("synthetic-B"))]}
        result=m.simulate(p)
        self.assertEqual(result["rejections"],0);self.assertEqual(result["final_state"]["applied"],m.key())
        p["events"]=[m.event(),m.event(None,m.key())]
        self.assertEqual(m.simulate(p)["steps"][-1]["error"],"first_nccl2_group_required")

    def test_version_rocm_backend_and_unknown_readiness(self):
        for changes in ({"compiled_nccl":[2,32,2]},{"rocm":True}):
            result=m.simulate({**m.DEFAULT,**changes,"events":[m.event(m.key())]})
            self.assertEqual(result["steps"][0]["error"],"compiled_version_or_rocm_gate")
        for backend in ("nccl-legacy","torchcomms"):
            result=m.simulate({**m.DEFAULT,"backend":backend,"events":[m.event(None,m.key()),m.event(m.key())]})
            self.assertFalse(result["steps"][0]["state"]["created_nccl2"])
            self.assertEqual(result["steps"][1]["error"],"explicit_option_unsupported_backend")
        for changes,expected in (({"tls_build":False},"blocked_tls_build"),({"runtime_libraries":False},"blocked_runtime_libraries"),
                                 ({"runtime_libraries":None},"requires_build_runtime_probe")):
            self.assertEqual(m.simulate({**m.DEFAULT,**changes,"events":[m.event(m.key())]})["readiness"],expected)

    def test_rejects_secret_fields_and_ambiguous_types_without_echo(self):
        p=copy.deepcopy(m.DEFAULT);p["events"][0]["option"]={"psk":"not-a-real-secret"}
        with self.assertRaisesRegex(ValueError,"metadata-only"):
            m.simulate(p)
        for field,value in (("compiled_nccl",[2,True,3]),("rocm",1),("tls_build",0)):
            with self.assertRaises(ValueError):
                m.simulate({**m.DEFAULT,field:value})
        p=copy.deepcopy(m.DEFAULT);p["events"][0]["option"]["cohort"]="not-a-real-secret"
        with self.assertRaises(ValueError) as error:
            m.simulate(p)
        self.assertNotIn("not-a-real-secret",str(error.exception))

    def test_cohort_metadata_agreement_matches_all_pairs(self):
        rng=random.Random(145)
        for n in range(65):
            labels=[rng.choice(("synthetic-A","synthetic-B")) for _ in range(n)]
            self.assertEqual(m.cohort_agreement(labels),m.pairwise_agreement(labels))
        self.assertTrue(m.cohort_agreement(["synthetic-A"]*512))
        self.assertFalse(m.cohort_agreement(["synthetic-A"]*511+["synthetic-B"]))

    def test_node_python_state_and_gate_parity(self):
        plans=[v["inputs"] for v in m.cases()]
        rng=random.Random(145)
        options=[None,m.key(),m.key("synthetic-B")]
        for _ in range(32):
            events=[m.event(rng.choice(options),rng.choice(options),api_ok=rng.choice((True,False))) for _ in range(8)]
            plans.append({**m.DEFAULT,"events":events})
        result=subprocess.run(["node"],input=JS+"\nconsole.log(JSON.stringify("+json.dumps(plans)+".map(simulate)));",capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[m.simulate(p) for p in plans])
        p=copy.deepcopy(m.DEFAULT);p["events"][0]["option"]={"psk":"not-a-real-secret"}
        result=subprocess.run(["node","-e",JS+"\nsimulate("+json.dumps(p)+");"],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn("metadata-only",result.stderr)

    def test_sealed_original_cpu_records(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency));self.assertIsNone(report["native_tls_verified"])
        matrix_card=cards(report).split('<section class="card">')[3]
        self.assertEqual(matrix_card.count("<tr>"),len(report["scenarios"])+1)
        for row in latency["rows"]:
            for name in ("set","pairwise"):
                samples=row[name+"_samples_ms"]
                self.assertEqual(len(samples),7);self.assertTrue(all(v>0 for v in samples))
                self.assertEqual(row[name+"_median_ms"],statistics.median(samples))


if __name__=="__main__":
    unittest.main()
