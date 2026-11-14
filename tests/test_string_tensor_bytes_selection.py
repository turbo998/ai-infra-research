import copy
import json
import random
import subprocess
import unittest

from experiments.string_tensor_bytes_selection import run as m
from experiments.string_tensor_bytes_selection.render import JS,cards


class StringBytesTests(unittest.TestCase):
    def test_original_wire_matches_independent_literal(self):
        r=m.evaluate(m.DEFAULT)
        expected=bytes.fromhex("0802080210083204007500003204e78cab003203fe41003200")
        self.assertEqual(bytes.fromhex(r["wire_hex"]),expected)
        self.assertEqual((r["payload_bytes"],r["framing_bytes"],r["wire_bytes"]),(11,14,25))
        self.assertEqual((r["preserved_elements"],r["naive_corrupted_elements"],r["naive_payload_bytes"]),(4,3,7))
        self.assertEqual(m.decode(expected),([2,2],[b"\0u\0\0","猫\0".encode(),b"\xfeA\0",b""]))

    def test_scalar_empty_is_not_zero_tensor(self):
        scalar=m.encode([],[b""]);zero=m.encode([0],[])
        self.assertEqual(scalar,bytes.fromhex("10083200"))
        self.assertEqual(zero,bytes.fromhex("08001008"))
        self.assertEqual(len(scalar),len(zero));self.assertNotEqual(scalar,zero)
        with self.assertRaises(ValueError):
            m.encode([],[])
        with self.assertRaises(ValueError):
            m.encode([0],[b""])

    def test_collisions_nuls_and_opaque_utf8(self):
        p={"dims":[3],"raw":False,"values":[{"kind":"bytes","hex":v} for v in ("73","7300","730000")]}
        r=m.evaluate(p)
        self.assertEqual((r["original_distinct_values"],r["naive_distinct_values"]),(3,1))
        self.assertEqual(r["naive_corrupted_elements"],2)
        row=m.evaluate(m.DEFAULT)["rows"][2]
        self.assertEqual(row["restored_hex"],"fe4100")
        self.assertIsNone(row["utf8_text"])
        self.assertEqual(row["utf8_status"],"invalid_utf8_bytes_preserved")

    def test_unicode_no_normalization_and_bom_retention(self):
        a=m.byte_value({"kind":"text","text":"é"});b=m.byte_value({"kind":"text","text":"e\u0301"})
        self.assertEqual((len(a),len(b)),(2,3));self.assertNotEqual(a,b)
        self.assertEqual(m.inspect_utf8(b"\xef\xbb\xbfz")["utf8_text"],"\ufeffz")
        with self.assertRaises(ValueError):
            m.byte_value({"kind":"text","text":"\ud800"})

    def test_varint_boundary_and_truncation(self):
        value=bytes(range(128));wire=m.encode([],[value])
        self.assertEqual(wire[:5],bytes.fromhex("1008328001"))
        self.assertEqual(m.decode(wire),([],[value]))
        for bad in (wire[:-1],b"\x10",b"\x10\x08\x32\x80",b"\x10\x08\x32\x80\x80",
                    b"\x10\x01\x32\x00",b"\x10\x08\x10\x08\x32\x00",b"\x10\x08\x4a\x00"):
            with self.subTest(wire=bad.hex()),self.assertRaises(ValueError):
                m.decode(bad)

    def test_invalid_input_explicit_errors(self):
        bad=[]
        for dims in ([True],[-1],[1.5],[17],[1,1,1]):
            p=copy.deepcopy(m.DEFAULT);p["dims"]=dims;bad.append(p)
        p=copy.deepcopy(m.DEFAULT);p["raw"]=True;bad.append(p)
        p=copy.deepcopy(m.DEFAULT);p["values"][0]={"kind":"bytes","hex":"0"};bad.append(p)
        p=copy.deepcopy(m.DEFAULT);p["values"][0]={"kind":"bytes","hex":"aa bb"};bad.append(p)
        p=copy.deepcopy(m.DEFAULT);p["values"][0]={"kind":"text","text":7};bad.append(p)
        p=copy.deepcopy(m.DEFAULT);p["values"].pop();bad.append(p)
        for p in bad:
            with self.subTest(input=p),self.assertRaises(ValueError):
                m.evaluate(p)

    def test_independent_random_wire_and_browser_parity(self):
        rng=random.Random(149);cases=[m.DEFAULT]
        for n in range(64):
            dims=[n%9];values=[bytes(rng.randrange(256) for _ in range(rng.randrange(18))) for _ in range(dims[0])]
            expected=b"\x08"+bytes(dims)+b"\x10\x08"+b"".join(b"\x32"+bytes([len(v)])+v for v in values)
            self.assertEqual(m.encode(dims,values),expected)
            self.assertEqual(m.decode(expected),(dims,values))
            cases.append({"dims":dims,"raw":False,"values":[{"kind":"bytes","hex":v.hex()} for v in values]})
        cases.append({"dims":[4],"raw":False,"values":[{"kind":"text","text":t} for t in ("猫\0","😀","e\u0301","\ufeffz")]})
        script=JS+"\nconsole.log(JSON.stringify("+json.dumps(cases,ensure_ascii=True)+".map(evaluate)));\n"
        output=subprocess.run(["node"],input=script,capture_output=True,text=True,encoding="utf-8",check=True)
        self.assertEqual(json.loads(output.stdout),[m.evaluate(p) for p in cases])

    def test_sealed_report_and_full_matrix(self):
        latency=json.loads((m.HERE/"latency.json").read_text())
        sealed=json.loads((m.HERE/"report.json").read_text())
        self.assertEqual(sealed,m.build_report(latency))
        self.assertFalse(sealed["independently_reproduced"])
        self.assertIsNone(sealed["native_legacy_numpy_outputs"])
        html=cards(sealed)
        self.assertEqual(html.count('class="card"'),7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",html)
        self.assertNotIn("<script",html)
        for row in sealed["case"]["outputs"]["rows"]:
            self.assertIn(row["kind"]+":"+row["hex"],html)
