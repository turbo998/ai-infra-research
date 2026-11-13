import json
import random
import subprocess
import unittest

from experiments.float_radix_contract_selection import run as m
from experiments.float_radix_contract_selection.render import JS


class FloatRadixTests(unittest.TestCase):
    def test_numeric_reference_and_raw_control(self):
        self.assertEqual(m.radix_order(m.FIXTURE),m.numeric_oracle(m.FIXTURE))
        raw=sorted(range(len(m.FIXTURE)),key=m.FIXTURE.__getitem__)
        self.assertNotEqual(raw,m.numeric_oracle(m.FIXTURE))

    def test_random_numeric_oracle_and_subnormals(self):
        rng=random.Random(4118)
        raw=[rng.getrandbits(32) for _ in range(500)]
        raw=[x for x in raw if x&0x7f800000!=0x7f800000 or not x&0x7fffff]
        raw += [0,0x80000000,1,0x80000001,0x007fffff,0x807fffff]
        for descending in (False,True):
            self.assertEqual(m.radix_order(raw,descending=descending),m.numeric_oracle(raw,descending))

    def test_stable_ties_and_reverse_output_control(self):
        for raw in (m.FIXTURE,list(reversed(m.FIXTURE))):
            for descending in (False,True):
                order=m.radix_order(raw,descending=descending)
                self.assertEqual(order,m.numeric_oracle(raw,descending))
                zeros=[i for i in order if raw[i]&0x7fffffff==0]
                self.assertEqual(zeros,sorted(zeros))
        self.assertNotEqual(list(reversed(m.radix_order(m.FIXTURE))),
                            m.numeric_oracle(m.FIXTURE,True))

    def test_partial_ranges_and_independent_bit_string_reference(self):
        for begin,end in ((0,1),(0,8),(3,19),(24,32),(0,32)):
            for descending in (False,True):
                keys=[]
                for raw in m.FIXTURE:
                    value=m.float_value(raw)
                    bits=f"{raw:032b}" if value!=0 else "0"*32
                    if value<0:
                        transformed="".join("1" if bit=="0" else "0" for bit in bits)
                    else:
                        transformed=("1" if bits[0]=="0" else "0")+bits[1:]
                    if descending:
                        transformed="".join("1" if bit=="0" else "0" for bit in transformed)
                    keys.append(int(transformed[32-end:32-begin],2))
                expected=sorted(range(len(keys)),key=keys.__getitem__)
                self.assertEqual(m.radix_order(m.FIXTURE,begin,end,descending),expected)
        self.assertNotEqual(m.radix_order(m.FIXTURE,0,8),
                            sorted(range(len(m.FIXTURE)),key=lambda i:m.FIXTURE[i]&255))

    def test_payload_and_empty_and_byte_bucket_stability(self):
        raw=list(m.FIXTURE)
        order=m.radix_order(raw)
        self.assertEqual(raw,m.FIXTURE)
        self.assertEqual(sorted(order),list(range(len(raw))))
        self.assertEqual(m.radix_order([]),[])
        for bits in (1,7,8,9,17,32):
            keys=[0,(1<<bits)-1,0,1,(1<<bits)-1]
            self.assertEqual(m.stable_lsd(keys,bits),sorted(range(len(keys)),key=keys.__getitem__))

    def test_explicit_invalid_inputs(self):
        for raw in (-1,2**32,1.0,True,0x7fc00000,0xff800001):
            with self.assertRaises(ValueError):
                m.sortable_key(raw)
        for begin,end in ((0,0),(-1,4),(0,33),(4,3),(True,8)):
            with self.assertRaises(ValueError):
                m.radix_order(m.FIXTURE,begin,end)
        for keys,bits in (([-1],8),([256],8),([True],8),([0],0)):
            with self.assertRaises(ValueError):
                m.stable_lsd(keys,bits)
        with self.assertRaises(ValueError):
            m.estimate_budget({**m.DEFAULT,"items":2**53-1})

    def test_node_hex_key_and_budget_parity(self):
        cases=[(m.FIXTURE,0,32,False),(m.FIXTURE,0,32,True),
               (m.FIXTURE,3,19,False),(list(reversed(m.FIXTURE)),0,8,True)]
        source=JS+"\nconst cases="+json.dumps(cases)+";\nconsole.log(JSON.stringify(cases.map(([raw,b,e,d])=>calculate(raw.map(x=>x.toString(16).padStart(8,'0')),b,e,d))));"
        actual=json.loads(subprocess.check_output(["node","-e",source],text=True))
        for result,(raw,begin,end,descending) in zip(actual,cases):
            self.assertEqual(result["order_ids"],m.radix_order(raw,begin,end,descending))
            self.assertEqual(result["selected_keys"],m.selected_keys(raw,begin,end,descending))
        source=JS+"\nconsole.log(JSON.stringify(estimate("+json.dumps(m.DEFAULT)+")));"
        self.assertEqual(json.loads(subprocess.check_output(["node","-e",source],text=True)),m.estimate_budget(m.DEFAULT))
        for call in ('calculate(["7fc00000"],0,32,false)','calculate(["00000000"],0,33,false)',
                     'estimate({items:1,value_bytes:2,begin_bit:0,end_bit:32})'):
            source=JS+f"\ntry{{{call};process.exit(1)}}catch(e){{console.log(e.message)}}"
            self.assertTrue(subprocess.check_output(["node","-e",source],text=True).strip())

    def test_sealed_report_and_timing_medians(self):
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("byte_pass","builtin"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],sorted(samples)[3])
        self.assertIsNone(report["native_temporary_storage_bytes"])


if __name__=="__main__":
    unittest.main()
