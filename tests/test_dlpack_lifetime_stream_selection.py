import itertools
import json
from pathlib import Path
import subprocess
import unittest

from experiments.dlpack_lifetime_stream_selection import run,render


class DLPackContractTests(unittest.TestCase):
    def test_strided_addresses_match_independent_recursive_walk(self):
        def walk(shape,strides,base,item):
            if not shape:
                return [base]
            return [address for i in range(shape[0])
                    for address in walk(shape[1:],strides[1:],base+i*strides[0]*item,item)]
        for rows,cols,pitch in itertools.product(range(4),range(1,4),range(3,6)):
            for item in (1,2,4,8):
                d=run.Descriptor((rows,cols),(pitch,1),8,item,256)
                self.assertEqual(d.addresses(),walk(d.shape,d.strides,8,item))
        reverse=run.Descriptor((3,),(-1,),8,4,12)
        self.assertEqual(reverse.addresses(),[8,4,0])
        self.assertEqual(reverse.bounds(),(0,12))
        with self.assertRaisesRegex(ValueError,"enumeration"):
            run.Descriptor((1048576,),(1,),0,1,1048576).addresses()

    def test_scalar_empty_legacy_and_versioned_stride_guards(self):
        self.assertEqual(run.Descriptor((),None,0,4,4).addresses(),[0])
        self.assertEqual(run.Descriptor((0,3),(3,1),0,4,0).addresses(),[])
        self.assertEqual(run.Descriptor((2,3),None,0,4,24,(1,1)).strides,(3,1))
        for args in [((2,3),None,0,4,24),((3,),(-1,),0,4,12),
                     ((2,3),(4,1),4,4,31),((2,3),(4,),0,4,64),
                     ((True,), (1,),0,4,64),((2,),(1,),0,3,64)]:
            with self.assertRaises(ValueError):
                run.Descriptor(*args)

    def test_fixture_exposes_byte_stride_error(self):
        r=run.build_report({"rows":[]})
        self.assertEqual(r["case"]["correct_values"],[1,2,3,5,6,7])
        self.assertEqual(r["case"]["addresses"],[4,8,12,20,24,28])
        self.assertEqual(r["case"]["wrong_value_count"],5)

    def test_one_shot_capsule_and_two_destruction_orders(self):
        for capsule_first in (True,False):
            owner=bytearray([3,5])
            cap=run.Capsule(owner)
            borrowed=cap.consume()
            with self.assertRaises(ValueError):
                cap.consume()
            if capsule_first:
                cap.close(); self.assertEqual(cap.deleter_calls,0)
                self.assertEqual(borrowed.value(1),5)
            borrowed.close(); cap.close(); borrowed.close()
            self.assertEqual(cap.deleter_calls,1)
            fresh=run.Capsule(owner)
            again=fresh.consume()
            self.assertEqual(again.value(0),3)
            again.close()
            with self.assertRaises(ValueError):
                borrowed.value(0)
        cap=run.Capsule(bytearray([1])); cap.close(); cap.close()
        self.assertEqual(cap.deleter_calls,1)

    def test_major_mismatch_release_precedes_any_payload_access(self):
        class Poison:
            def __getitem__(self,index):
                raise AssertionError("major-mismatched payload interpreted")
        cap=run.Capsule(Poison(),major=2)
        with self.assertRaisesRegex(ValueError,"major mismatch"):
            cap.consume()
        cap.close()
        self.assertEqual(cap.deleter_calls,1)

    def test_readonly_does_not_make_snapshot_and_copy_is_distinct(self):
        owner=bytearray([3,5])
        alias,copy=run.Capsule(owner,readonly=True),run.Capsule(owner,copy=True,readonly=True)
        a,b=alias.consume(),copy.consume()
        owner[0]=9
        self.assertEqual((a.value(0),b.value(0)),(9,3))
        self.assertFalse(alias.copied); self.assertTrue(copy.copied)
        with self.assertRaises(ValueError):
            a.set(0,7)
        a.close(); b.close()

    def test_stream_schedule_enumeration_matches_full_permutation_oracle(self):
        all_sequences=[s for s in itertools.permutations(("write","record","wait","read"))
                       if s.index("write")<s.index("record") and s.index("wait")<s.index("read")]
        self.assertEqual(set(run.schedules(False)),set(all_sequences))
        guarded=[s for s in all_sequences if s.index("record")<s.index("wait")]
        self.assertEqual(set(run.schedules()),set(guarded))
        self.assertEqual(len(all_sequences),6)
        self.assertEqual(sum(s.index("read")<s.index("write") for s in all_sequences),1)
        self.assertTrue(all(s.index("write")<s.index("read") for s in guarded))

    def test_budget_javascript_parity_and_sealed_contracts(self):
        profiles=[run.DEFAULT,{**run.DEFAULT,"rows":0,"owner_bytes":0},
                  {**run.DEFAULT,"rows":1,"columns":3,"row_stride_elements":3,"element_bytes":8},
                  {**run.DEFAULT,"rows":2,"columns":3,"row_stride_elements":4,"byte_offset":4,"element_bytes":4,"owner_bytes":32}]
        script="const DEFAULT="+json.dumps(run.DEFAULT)+";\n"+render.JS
        script+="console.log(JSON.stringify("+json.dumps(profiles)+".map(estimate)));"
        script+="for(const p of [{rows:-1},{element_bytes:3},{row_stride_elements:2},{owner_bytes:10},{rows:2**53-1}]){let ok=false;try{estimate({...DEFAULT,...p})}catch(e){ok=true}if(!ok)throw Error('invalid accepted')}"
        output=subprocess.run(["node","-e",script],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(output.stdout),[run.estimate_budget(p) for p in profiles])
        for patch in ({"rows":-1},{"element_bytes":3},{"row_stride_elements":2},{"owner_bytes":10},{"rows":2**53-1}):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,**patch})
        report=json.loads((run.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(run.build_report(report["latency"]),report)
        cards=render.cards(report)
        self.assertEqual(cards.count('<section class="card">'),7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",cards)
        self.assertNotIn("<script",cards)
        self.assertEqual(cards.count("<table>"),cards.count("</table>"))


if __name__=="__main__":
    unittest.main()
