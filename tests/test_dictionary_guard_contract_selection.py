import json
import random
import subprocess
import unittest

from experiments.dictionary_guard_contract_selection import run as m
from experiments.dictionary_guard_contract_selection.render import JS,cards


class DictionaryGuardTests(unittest.TestCase):
    def test_named_reuses_order_does_not(self):
        p={**m.DEFAULT,"program":"named"};named=m.evaluate(p);order=m.evaluate(m.DEFAULT)
        self.assertEqual((named["guarded_plans"],named["naive_misses"]),(1,0))
        self.assertEqual([r["guarded"]["value"] for r in named["rows"]],[28,28])
        self.assertEqual((order["guarded_plans"],order["naive_misses"]),(2,1))
        self.assertEqual(order["rows"][1]["expected"]["order"],["beta","alpha"])

    def test_binding_error_identity_requires_order(self):
        for p in m.suite_inputs():
            if p["program"] in ("unexpected","duplicate"):
                r=m.evaluate(p)
                self.assertEqual(r["naive_misses"],1)
                self.assertNotEqual(r["rows"][0]["expected"]["error_key"],r["rows"][1]["expected"]["error_key"])
                self.assertIsNone(r["rows"][1]["expected"]["value"])

    def test_native_protocol_length_and_type(self):
        native=m.native_oracle()
        self.assertEqual(native["mapping_protocol"],{"value":73,"events":["keys","get:alpha"],"storage_shortcut_value":28})
        self.assertEqual(native["length_contract"],{"user_len":77,"structural_dict_len":2})
        self.assertEqual(native["type_contract"],{"isinstance_set":True,"actual_type_is_dict_subclass":True})
        self.assertIsNone(native["native_torch_compile"])

    def test_model_matches_independent_native_observations(self):
        native=m.native_oracle()["observations"]
        for p in m.suite_inputs():
            modeled=m.evaluate(p)
            facts=[v["output"] for v in native if v["program"]==p["program"]]
            for row,fact in zip(modeled["rows"],facts):
                for key,value in fact.items():
                    self.assertEqual(row["expected"][key],value)
            self.assertEqual(modeled["guarded_misses"],0)

    def test_dynamic_scalar_model_and_type_gate(self):
        calls=[{**m.DEFAULT["calls"][0]},{"mapping_type":"dict","pairs":[["alpha",4],["beta",-1]],"base":2}]
        r=m.evaluate({"program":"named","calls":calls})
        self.assertEqual(r["guarded_plans"],1);self.assertEqual(r["rows"][1]["guarded"]["value"],41)
        p=next(p for p in m.suite_inputs() if p["program"]=="type")
        r=m.evaluate(p);self.assertEqual(r["naive_misses"],1)

    def test_invalid_inputs_and_guard_length_boundaries(self):
        for call in ({"mapping_type":"dict","pairs":[["alpha",True]],"base":3},
                     {"mapping_type":"dict","pairs":[["alpha",1],["alpha",2]],"base":3},
                     {"mapping_type":"fake","pairs":[],"base":3}):
            with self.assertRaises(ValueError):
                m.evaluate({"program":"named","calls":[call]})
        expected=("a","b")
        for scalar in (False,True):
            self.assertTrue(m.order_matches(expected,["a","b"],scalar))
            self.assertFalse(m.order_matches(expected,["b","a"],scalar))
            self.assertFalse(m.order_matches(expected,["a"],scalar))
            self.assertFalse(m.order_matches(expected,["a","b","c"],scalar))

    def test_random_order_oracle_and_browser_parity(self):
        rng=random.Random(152);inputs=m.suite_inputs()
        for _ in range(32):
            keys=["field0","field1","field2","field3"];rng.shuffle(keys)
            pairs=[[key,i] for i,key in enumerate(keys)]
            p={"program":"order","calls":[{"mapping_type":"dict","pairs":pairs,"base":1},
               {"mapping_type":"dict","pairs":list(reversed(pairs)),"base":2}]}
            r=m.evaluate(p);self.assertEqual(r["rows"][1]["guarded"]["order"],list(reversed(keys)));inputs.append(p)
        output=subprocess.run(["node"],input=JS+"\nconsole.log(JSON.stringify("+json.dumps(inputs)+".map(evaluate)));",
                              capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(output.stdout),[m.evaluate(p) for p in inputs])

    def test_sealed_native_facts_and_full_matrix(self):
        latency=json.loads((m.HERE/"latency.json").read_text());native=json.loads((m.HERE/"native_oracle.json").read_text())
        sealed=json.loads((m.HERE/"report.json").read_text())
        self.assertEqual(sealed,m.build_report(latency,native))
        self.assertEqual(native["observations"],m.native_oracle()["observations"])
        self.assertEqual(sealed["summary"],{"calls":12,"guarded_plans":11,"naive_plans":6,"guarded_misses":0,"naive_misses":5})
        html=cards(sealed);self.assertEqual(html.count('class="card"'),7);self.assertEqual(html.count("✕ Key-set-only stale"),5)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",html);self.assertNotIn("<script",html)
