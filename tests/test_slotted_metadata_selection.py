from dataclasses import asdict
import json
import random
import subprocess
import unittest
from experiments.slotted_metadata_selection.run import DEFAULT,CLASSES,layout_probe,plan,validate
from experiments.slotted_metadata_selection.render import JS


class SlottedMetadataTests(unittest.TestCase):
    def test_fields_pickle_and_canonical_shell_lookup(self):
        a=plan(DEFAULT);b=plan({**DEFAULT,"layout":"plain"})
        self.assertEqual(a["operation"],b["operation"])
        self.assertLess(a["measured_canonical_batch_shell_bytes"],b["measured_canonical_batch_shell_bytes"])
        self.assertIsNone(a["referent_bytes"]);self.assertIsNone(a["native_vllm_verified"])

    def test_explicit_dictionary_contract_errors(self):
        for op,error in (("vars","TypeError"),("dynamic","AttributeError")):
            a=plan({**DEFAULT,"operation":op})
            self.assertEqual(a["operation"],{"status":"expected_contract_error","error":error})
            self.assertEqual(plan({**DEFAULT,"layout":"inherited_dict","operation":op})["operation"]["status"],"ok")
        self.assertEqual(plan({**DEFAULT,"layout":"inherited_dict","operation":"vars"})["operation"]["value"],{})

    def test_weakref_optin_and_gc_scope(self):
        self.assertEqual(plan({**DEFAULT,"operation":"weakref"})["operation"]["error"],"TypeError")
        self.assertTrue(plan({**DEFAULT,"layout":"weak_slots","operation":"weakref"})["operation"]["alive"])
        self.assertTrue(plan(DEFAULT)["gc_tracked"])

    def test_record_operations_preserve_bounded_payload(self):
        p={**DEFAULT,"rid":1048576,"work":0,"payload":[0,1048576]}
        for name in CLASSES:
            self.assertEqual(plan({**p,"layout":name})["operation"]["value"],{"rid":p["rid"],"work":0,"payload":p["payload"]})

    def test_live_probe_compatibility_and_aliasing(self):
        a=layout_probe()
        self.assertEqual(a["cached_property_without_dict"],"TypeError")
        self.assertTrue(a["pickle_preserves_shared_payload"])
        self.assertTrue(a["slotted_declared_fields_remain_mutable"])
        self.assertFalse(a["profiles"]["slots"]["has_instance_dict"]);self.assertTrue(a["profiles"]["inherited_dict"]["has_instance_dict"])

    def test_invalid_unmeasured_or_unknown_profile(self):
        for k,v in (("records",65),("records",True),("layout","native"),("payload",[-1]),("operation","rss")):
            with self.assertRaises(ValueError):validate({**DEFAULT,k:v})
        with self.assertRaises(ValueError):validate({**DEFAULT,"native_bytes":0})

    def test_sealed_shell_tiers_are_exact_batches(self):
        from experiments.slotted_metadata_selection.run import HERE
        rows=json.loads((HERE/"latency.json").read_text())["rows"]
        for row in rows:
            self.assertGreater(row["shell_bytes"]["plain"],row["shell_bytes"]["slots"])
            self.assertGreater(row["shell_bytes"]["weak_slots"],row["shell_bytes"]["slots"])
        self.assertEqual([r["records"] for r in rows],[64,256,1024,4096])

    def test_node_65contract_predictions(self):
        from experiments.slotted_metadata_selection.run import HERE
        data={"latency":json.loads((HERE/"latency.json").read_text()),"layout_probe":layout_probe()}
        rng=random.Random(165);cases=[DEFAULT]
        for _ in range(64):
            cases.append({**DEFAULT,"layout":rng.choice(list(CLASSES)),"operation":rng.choice(["fields","vars","dynamic","weakref","pickle"]),
                          "records":rng.choice([64,256,1024,4096]),"rid":rng.randrange(256),"work":rng.randrange(1024),"payload":[rng.randrange(64) for _ in range(rng.randrange(8))]})
        script=JS+'const fs=require("fs");const x=JSON.parse(fs.readFileSync(0,"utf8"));console.log(JSON.stringify(x.cases.map(p=>plan(p,x.data))));'
        result=subprocess.run(["node","-e",script],input=json.dumps({"cases":cases,"data":data}),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout),[plan(p) for p in cases])


if __name__=="__main__":unittest.main()
