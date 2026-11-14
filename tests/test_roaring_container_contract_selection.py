import json
import random
import statistics
import struct
import subprocess
import unittest

from experiments.roaring_container_contract_selection import run as m
from experiments.roaring_container_contract_selection.render import JS


class RoaringContainerTests(unittest.TestCase):
    def test_nonrun_array_bitset_thresholds_and_actual_bytes(self):
        for n,kind in ((4096,"array"),(4097,"bitset")):
            ids=list(range(0,n*2,2));c=m.encode(ids,False)
            self.assertEqual(c[0]["kind"],kind)
            self.assertEqual(len(bytes.fromhex(c[0]["payload_hex"])),8192)
            self.assertEqual(m.decode(c),ids)

    def test_contiguous_full_low16_run_length_minus_one(self):
        ids=list(range(65536));c=m.encode(ids)
        self.assertEqual(c[0]["kind"],"run")
        self.assertEqual(bytes.fromhex(c[0]["payload_hex"]),struct.pack("<HHH",1,0,65535))
        self.assertEqual(m.decode(c),ids)

    def test_roundtrip_high_keys_unsigned_and_empty(self):
        ids=[0,65535,65536,2**31,2**32-1]
        self.assertEqual(m.decode(m.encode(ids+ids[::-1])),ids)
        self.assertEqual(m.encode([]),[])
        self.assertEqual(m.decode([]),[])
        self.assertEqual(m.intersection_sorted([7],[65543]),[])

    def test_independent_interval_oracle_and_set_algebra(self):
        rng=random.Random(137)
        for _ in range(24):
            a=rng.sample(range(256),40)+list(range(65536+10,65536+30))
            b=rng.sample(range(256),30)+list(range(65536+20,65536+40))
            left=m.decode(m.encode(a));right=m.decode(m.encode(b))
            got=m.intersection_sorted(left,right)
            self.assertEqual(got,m.interval_intersection_oracle(a,b))
            self.assertEqual(got,sorted(set(a)&set(b)))
            self.assertEqual(got,m.intersection_sorted(right,left))

    def test_duplicate_order_and_run_policy(self):
        a,_=m.fixtures()
        self.assertEqual(m.describe(a),m.describe(a[::-1]+a))
        self.assertEqual(m.describe(a)["container_payload_bytes"],12)
        self.assertEqual(m.describe(a,False)["container_payload_bytes"],206)
        self.assertEqual(m.encode([1,2,3])[0]["kind"],"array")

    def test_malformed_payloads_fail_explicitly(self):
        for ids in ([-1],[2**32],[True],[1.5]):
            with self.assertRaises(ValueError):
                m.encode(ids)
        c=m.encode(range(10));c[0]["payload_hex"]="0000"
        with self.assertRaises(ValueError):
            m.decode(c)
        c=m.encode(range(10));c[0]["payload_hex"]=struct.pack("<HHH",1,65535,1).hex()
        with self.assertRaises(ValueError):
            m.decode(c)
        c=m.encode([1,3]);c[0]["payload_hex"]=struct.pack("<HH",3,1).hex()
        with self.assertRaises(ValueError):
            m.decode(c)

    def test_node_python_summary_and_exact_intersection_parity(self):
        a,b=m.fixtures()
        out=json.loads(subprocess.check_output(["node","-e",JS+"\nconsole.log(JSON.stringify(simulate("+json.dumps(a)+","+json.dumps(b)+")))"],text=True))
        self.assertEqual(out["A"],m.describe(a))
        self.assertEqual(out["B"],m.describe(b))
        self.assertEqual(out["intersection"],sorted(set(a)&set(b)))
        self.assertEqual(out["intersection_count"],52)

    def test_sealed_report_and_timing_records(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("integer_and","sorted_merge"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(256);self.assertEqual(a(),b())
        self.assertIsNone(report["native_roaring64_rank_select"])


if __name__=="__main__":
    unittest.main()
