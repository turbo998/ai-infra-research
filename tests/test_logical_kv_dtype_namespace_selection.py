import copy
from fractions import Fraction
import json
import random
import struct
import subprocess
import unittest

from experiments.logical_kv_dtype_namespace_selection import run as m
from experiments.logical_kv_dtype_namespace_selection.render import JS,cards


class LogicalDtypeTests(unittest.TestCase):
    def test_independent_scalar_probes(self):
        self.assertEqual(m.probe(b"\x3c","float8_e4m3fn")["ratio"],{"numerator":"3","denominator":"2"})
        self.assertEqual(m.probe(b"\x3c","float8_e5m2")["ratio"],{"numerator":"1","denominator":"1"})
        self.assertEqual(m.probe(bytes.fromhex("803f"),"bfloat16")["ratio"],{"numerator":"1","denominator":"1"})
        self.assertEqual(m.probe(bytes.fromhex("003e"),"float16")["ratio"],{"numerator":"3","denominator":"2"})

    def test_namespace_objects_and_foreign_witnesses(self):
        r=m.evaluate(m.DEFAULT)
        self.assertEqual((r["legacy_objects"],r["physical_only_objects"],r["typed_objects"]),(1,3,4))
        self.assertEqual((r["legacy_foreign_reads"],r["physical_only_foreign_reads"],r["typed_foreign_reads"]),(3,1,0))
        self.assertEqual(r["selected_stored_payload_bytes"],{"legacy":2,"physical_only":6,"typed":8})
        self.assertEqual(r["rows"][0]["physical_only"]["finite_probe_absolute_error"],{"numerator":"1","denominator":"2"})

    def test_cold_migration_never_falls_back(self):
        r=m.evaluate(m.DEFAULT)["migration"]
        self.assertTrue(r["legacy_exists"]);self.assertFalse(r["fallback_used"])
        self.assertEqual(r["typed_before_repopulate"]["status"],"miss")
        self.assertIsNone(r["typed_before_repopulate"]["payload_hex"])
        self.assertEqual(r["typed_after_repopulate"]["payload_hex"],"3c00")

    def test_storage_extra_tags_and_missing_model(self):
        original=m.key(m.DEFAULT,"logical:float8_e4m3fn")
        for change in ({"storage_tag":"envelope-v2"},{"extra_backend_tag":"layout-v2"},{"model_id":None}):
            self.assertNotEqual(original,m.key({**m.DEFAULT,**change},"logical:float8_e4m3fn"))
        p={**m.DEFAULT,"model_id":None,"storage_tag":None}
        self.assertIn("logical:float8_e4m3fn",m.evaluate(p)["rows"][0]["typed_key"])

    def test_invalid_unresolved_dtype_and_payload(self):
        for change in ({"logical_dtype":None},{"logical_dtype":"auto"},{"payload_hex":"0"},{"payload_hex":"aa bb"}):
            p=copy.deepcopy(m.DEFAULT);p["writes"][0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.evaluate(p)
        p=copy.deepcopy(m.DEFAULT);p["writes"][2]["payload_hex"]="00"
        with self.assertRaises(ValueError):
            m.evaluate(p)

    def test_signed_zero_nonfinite_and_native_ieee_half_oracle(self):
        self.assertTrue(m.probe(b"\x80","float8_e4m3fn")["negative_zero"])
        self.assertEqual(m.probe(b"\x7e","float8_e4m3fn")["ratio"],{"numerator":"448","denominator":"1"})
        self.assertEqual(m.probe(b"\x7f","float8_e4m3fn")["status"],"nan")
        self.assertEqual(m.probe(b"\x7c","float8_e5m2")["status"],"infinity")
        rng=random.Random(153)
        for _ in range(64):
            word=rng.randrange(0x7c00);data=word.to_bytes(2,"little")
            value=Fraction.from_float(struct.unpack("<e",data)[0])
            self.assertEqual(m.probe(data,"float16")["ratio"],m.ratio(value))

    def test_node_parity_and_hoisted_keys(self):
        inputs=[m.DEFAULT,{**m.DEFAULT,"model_id":None,"storage_tag":"envelope-v2"}]
        for dtype in m.FORMATS:
            inputs.append({**m.DEFAULT,"writes":[w for w in m.DEFAULT["writes"] if w["logical_dtype"]==dtype]})
        script=JS+"\nconsole.log(JSON.stringify("+json.dumps(inputs)+".map(evaluate)));"
        result=subprocess.run(["node"],input=script,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[m.evaluate(p) for p in inputs])
        for n in (1,8,64):
            funcs=m.timing_paths(n);self.assertEqual(funcs[0](),funcs[1]())

    def test_sealed_report_and_matrix(self):
        latency=json.loads((m.HERE/"latency.json").read_text());sealed=json.loads((m.HERE/"report.json").read_text())
        self.assertEqual(sealed,m.build_report(latency));self.assertFalse(sealed["independently_reproduced"])
        self.assertIsNone(sealed["native_sglang_verified"])
        html=cards(sealed);self.assertEqual(html.count('class="card"'),7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",html);self.assertNotIn("<script",html)
        for dtype in m.FORMATS:
            self.assertIn(dtype,html)
