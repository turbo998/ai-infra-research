import hashlib
from itertools import permutations
import json
import statistics
import subprocess
import unittest

from experiments.multipart_completion_contract_selection import run as m
from experiments.multipart_completion_contract_selection.render import JS


class MultipartCompletionTests(unittest.TestCase):
    def test_out_of_order_upload_and_independent_assembly(self):
        for order in permutations((1,2,3)):
            u=m.Upload();tokens={n:u.put(n,bytes([n])*n) for n in order}
            manifest=[[n,tokens[n]] for n in (1,2,3)]
            expected=m.independent_assembly(u.parts,manifest)
            self.assertIsNone(u.object)
            self.assertEqual(u.complete(manifest),expected)
            self.assertFalse(u.active);self.assertEqual(u.parts,{})

    def test_replacement_stale_token_does_not_publish(self):
        u=m.Upload();old=u.put(1,b"old");new=u.put(1,b"new")
        with self.assertRaisesRegex(ValueError,"stale"):
            u.complete([[1,old]])
        self.assertIsNone(u.object);self.assertTrue(u.active)
        self.assertEqual(u.complete([[1,new]]),b"new")

    def test_supplied_subset_and_expected_length_guard(self):
        u=m.Upload();a=u.put(1,b"A");u.put(2,b"B")
        with self.assertRaisesRegex(ValueError,"length"):
            u.complete([[1,a]],expected_length=2)
        self.assertIsNone(u.object)
        self.assertEqual(u.complete([[1,a]]),b"A")

    def test_ascending_unique_consecutive_and_checksum(self):
        u=m.Upload();a=u.put(1,b"A");c=u.put(3,b"C")
        for manifest in ([[3,c],[1,a]],[[1,a],[1,a]],[[1,a],[3,c]]):
            with self.assertRaises(ValueError):
                u.complete(manifest)
        with self.assertRaisesRegex(ValueError,"SHA256"):
            u.complete([[1,a],[3,c]],expected_sha256=hashlib.sha256(b"wrong").hexdigest(),consecutive=False)
        self.assertEqual(u.complete([[1,a],[3,c]],consecutive=False),b"AC")

    def test_http200_error_body_is_not_success(self):
        with self.assertRaisesRegex(ValueError,"not a success"):
            m.parse_completion(200,"<Error><Code>InternalError</Code></Error>")
        with self.assertRaises(ValueError):
            m.parse_completion(500,"<Error/>")
        self.assertEqual(m.parse_completion(200,'<CompleteMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><ETag>x</ETag></CompleteMultipartUploadResult>')["kind"],"completion_result")

    def test_native_size_plan_limits(self):
        self.assertEqual(m.budget(m.DEFAULT)["selected_part_count"],8192)
        self.assertEqual(m.budget({**m.DEFAULT,"object_bytes":1})["selected_final_part_bytes"],1)
        for p in ({**m.DEFAULT,"part_bytes":1024},{**m.DEFAULT,"part_bytes":6*1024**3},{**m.DEFAULT,"object_bytes":2**53}):
            with self.assertRaises(ValueError):
                m.budget(p)

    def test_node_state_and_plan_parity(self):
        source=JS+"\nconsole.log(JSON.stringify({result:simulate([[2,'BBBB'],[1,'AAAA'],[1,'CCCC'],[3,'DD']],[[1,'opaque-revision-3'],[2,'opaque-revision-1'],[3,'opaque-revision-4']],true,10),budget:budget("+json.dumps(m.DEFAULT)+")}))"
        out=json.loads(subprocess.check_output(["node","-e",source],text=True))
        self.assertEqual(out["result"]["assembled_ascii"],"CCCCBBBBDD")
        self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_timing_paths(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("join","copy"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(16);self.assertEqual(a(),b())
        self.assertIsNone(report["checkpoint_global_atomicity"])


if __name__=="__main__":
    unittest.main()
