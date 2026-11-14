import json
import random
import subprocess
import unittest

from experiments.sqlite_page_sentinel_selection import run as m
from experiments.sqlite_page_sentinel_selection.render import JS


class SqlitePageSentinelTests(unittest.TestCase):
    def test_field_specific_sentinels_and_negative_naive_formula(self):
        r=m.layout(m.DEFAULT)
        self.assertEqual(m.decode_size(1),65536)
        self.assertEqual(r["decoded_content_offset"],65536)
        self.assertEqual(r["unused_without_freeblocks_bytes"],65528)
        self.assertEqual(r["naive_raw_offset_unused_bytes"],-8)
        data,p=m.opaque_leaf(65536,[])
        decoded=m.decode_leaf(data,2)
        self.assertEqual(decoded["first_freeblock_raw"],0);self.assertEqual(p["cells"],0)

    def test_independent_space_conservation_oracle(self):
        rng=random.Random(143)
        for _ in range(40):
            size=rng.choice([4096,8192,65536]);first=bool(rng.randrange(2));reserved=rng.choice([0,16,32])
            lengths=[rng.randint(4,64) for _ in range(rng.randrange(8))]
            data,p=m.opaque_leaf(size,lengths,first,reserved);r=m.decode_leaf(data,1 if first else 2,reserved)["outputs"]
            used=(100 if first else 0)+8+2*len(lengths)+sum(lengths)
            self.assertEqual(r["unused_without_freeblocks_bytes"]+used+reserved,size)

    def test_first_page_and_reserved_bytes_change_geometry(self):
        first=m.layout({**m.DEFAULT,"first_page":True})
        reserved=m.layout({**m.DEFAULT,"reserved_bytes":32,"content_raw":65504})
        self.assertEqual(first["unused_without_freeblocks_bytes"],65428)
        self.assertEqual(reserved["unused_without_freeblocks_bytes"],65496)
        with self.assertRaises(ValueError):
            m.layout({**m.DEFAULT,"reserved_bytes":32})

    def test_nonempty_content_offset_is_not_zero_sentinel(self):
        data,p=m.opaque_leaf(65536,[9,17,33]);r=m.decode_leaf(data,2)["outputs"]
        self.assertEqual(p["content_raw"],65477)
        self.assertEqual(r["pointer_array_bytes"],6)
        self.assertEqual(r["unused_without_freeblocks_bytes"],65463)

    def test_actual_sqlite_raw_header_oracle_and_missing_metrics(self):
        fixture=m.sqlite_fixture()
        self.assertEqual([r["file_page_size_raw"] for r in fixture["rows"]],[4096,1])
        self.assertEqual([r["decoded"]["inputs"]["content_raw"] for r in fixture["rows"]],[4096,0])
        self.assertEqual([r["decoded"]["outputs"]["unused_without_freeblocks_bytes"] for r in fixture["rows"]],[4088,65528])
        for row in fixture["rows"]:
            if not row["dbstat_available"]:
                self.assertIsNone(row["native_dbstat_unused_bytes"])
                self.assertIn("unavailable",row["dbstat_unavailable_reason"])

    def test_invalid_page_and_header_states_are_explicit(self):
        for raw in (0,2,65535,True):
            with self.assertRaises(ValueError):
                m.decode_size(raw)
        for key,value in (("page_bytes",4096),("first_page",1),("cells",-1),("fragmented_bytes",61),("content_raw",4)):
            with self.assertRaises(ValueError):
                m.layout({**m.DEFAULT,key:value})
        data,_=m.opaque_leaf(4096,[]);bad=bytearray(data);bad[1:3]=b"\0\x10"
        with self.assertRaises(ValueError):
            m.decode_leaf(bytes(bad),2)
        with self.assertRaises(ValueError):
            m.layout({**m.DEFAULT,"page_bytes":512,"reserved_bytes":255,"content_raw":257})

    def test_node_python_geometry_parity_and_bad_input(self):
        for p in (m.DEFAULT,{**m.DEFAULT,"first_page":True},{**m.DEFAULT,"reserved_bytes":32,"content_raw":65504}):
            result=subprocess.run(["node","-e",JS+"\nconsole.log(JSON.stringify(layout("+json.dumps(p)+")));"],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout),m.layout(p))
        result=subprocess.run(["node","-e",JS+"\nlayout("+json.dumps({**m.DEFAULT,"page_bytes":4096})+");"],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)

    def test_sealed_report_and_original_cpu_paths(self):
        for n in (64,256,1024,4096):
            a,b=m.timing_paths(n);self.assertEqual(a(),b())
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        for row in latency["rows"]:
            for path in ("unpacked","manual"):
                samples=row[path+"_samples_ms"];self.assertEqual(len(samples),7)
                self.assertEqual(sorted(samples)[3],row[path+"_median_ms"])
        native=json.loads((m.HERE/"native_oracle.json").read_text(encoding="utf-8"))
        self.assertEqual(json.loads((m.HERE/"report.json").read_text(encoding="utf-8")),m.build_report(latency,native))
