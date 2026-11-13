import json
import re
import statistics
import struct
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.arrow_nullable_slice_selection import render, run


class ArrowNullableSliceTests(unittest.TestCase):
    def test_logical_null_empty_and_child_null_are_distinct(self):
        s = run.simulate()["slice"]
        self.assertEqual(s["logical_rows"],[None,[],[20,None],[30,31]])
        self.assertEqual(s["maximum_row_mismatches"],0)
        self.assertEqual(s["visible_offsets"],[2,4,4,6,8])
        self.assertEqual(s["logical_flatten"],[20,None,30,31])
        self.assertEqual(s["contiguous_child_span"],[99,98,20,None,30,31])
        self.assertNotEqual(s["values_property"],s["logical_flatten"])

    def test_all_slices_nested_offsets_and_byte_crossings(self):
        parent = run.fixture()
        expected = [[10,11],None,[],[20,None],[30,31],None]
        for start in range(7):
            for length in range(7-start):
                view = run.ListView(parent,start,length)
                self.assertEqual(view.logical_rows(),expected[start:start+length])
                self.assertEqual(run.compact(view).logical_rows(),view.logical_rows())
        whole = run.ListView(parent,0,6)
        self.assertEqual(whole.slice(1,4).slice(1,2).logical_rows(),expected[2:4])
        rows = [None if i%3==0 else [i] for i in range(19)]
        larger = run.canonical_buffers(rows)
        self.assertEqual(run.ListView(larger,7,10).logical_rows(),rows[7:17])
        with self.assertRaises(ValueError):
            whole.slice(1,4).slice(4,1)

    def test_offset_negative_controls(self):
        r = run.simulate()
        self.assertEqual(r["controls"]["wrong_bit_row_mismatches"],2)
        self.assertEqual(r["controls"]["wrong_child_row_mismatches"],2)
        self.assertNotEqual(r["controls"]["ignore_parent_row_offset"],r["slice"]["logical_rows"])

    def test_required_objects_nullable_buffers_and_storage_validation(self):
        child = run.ChildInt32(struct.pack("<i",5))
        all_valid = run.ListBuffers(struct.pack("<ii",0,1),child,None)
        self.assertEqual(run.ListView(all_valid,0,1).logical_rows(),[[5]])
        with self.assertRaises(TypeError):
            run.ListBuffers(struct.pack("<ii",0,1),None)
        with self.assertRaises(TypeError):
            run.ListView(None,0,0)
        with self.assertRaises(ValueError):
            run.ChildInt32(struct.pack("<ii",1,2),b"")
        for offsets in ((0,2),(1,0),(-1,0)):
            with self.assertRaises(ValueError):
                run.ListBuffers(struct.pack("<ii",*offsets),child)
        empty = run.canonical_buffers([])
        self.assertEqual(run.ListView(empty,0,0).logical_rows(),[])

    def test_view_shares_parent_while_canonical_copy_rebases_buffers(self):
        parent = run.fixture()
        view = run.ListView(parent,1,4)
        copied = run.compact(view)
        self.assertIs(view.parent,parent)
        self.assertIs(view.parent.child.data,parent.child.data)
        self.assertIsNot(copied.parent,parent)
        self.assertEqual([copied.parent.position(i) for i in range(5)],[0,0,0,2,4])
        self.assertEqual((parent.buffer_bytes,copied.parent.buffer_bytes),(67,38))
        self.assertEqual(copied.logical_rows(),view.logical_rows())

    def test_exact_budget_bit_ceiling_and_offset_width(self):
        b = run.estimate_budget(run.DEFAULT)
        self.assertEqual(b["view_retained_buffer_bytes"],38928388)
        self.assertEqual(b["canonical_copy_buffer_bytes"],1126)
        odd = run.estimate_budget({**run.DEFAULT,"slice_rows":9,"visible_child_values":9})
        self.assertEqual(odd["canonical_copy_buffers"]["parent_validity"],2)
        self.assertEqual(odd["canonical_copy_buffers"]["child_validity"],2)
        wide = {**run.DEFAULT,"child_values":2**31,"visible_child_values":0}
        self.assertFalse(run.estimate_budget(wide)["max_offset_fits_selected_signed_type"])
        self.assertTrue(run.estimate_budget({**wide,"offset_bytes":8})["max_offset_fits_selected_signed_type"])
        for key,value in (("value_bytes",True),("slice_rows",2**31),("parent_bitmap",True),
                          ("parent_rows",2**53),("slice_rows",0)):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,key:value})

    def test_sealed_artifacts_csp_svg_and_measured_sample_shape(self):
        for path,content in run.outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
        r = run.simulate()
        self.assertEqual(r["latency"]["operations_per_repeat"],25)
        for row in r["latency"]["rows"]:
            self.assertEqual(len(row["primary_samples_ms"]),7)
            self.assertEqual(row["primary_median_ms"],statistics.median(row["primary_samples_ms"]))
            self.assertEqual(row["baseline_median_ms"],statistics.median(row["baseline_samples_ms"]))
            self.assertEqual(row["logical_mismatches"],0)
        deck = render.deck(r)
        self.assertEqual(deck.count('<section class="card"'),7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",deck)
        self.assertNotIn("<script",deck)
        self.assertNotRegex(deck,r'(?:src|href)="https?://')
        svg = render.summary_svg(r)
        root = ET.fromstring(svg)
        self.assertEqual((root.attrib["width"],root.attrib["height"]),("1600","1000"))
        for points in re.findall(r'<polygon points="([^"]+)',svg):
            for point in points.split():
                y = float(point.split(",")[1])
                self.assertGreaterEqual(y,200)
                self.assertLessEqual(y,620)

    def test_javascript_exact_byte_and_bitmap_parity(self):
        profiles = [run.DEFAULT,{**run.DEFAULT,"parent_bitmap":0,"child_bitmap":0},
                    {**run.DEFAULT,"slice_rows":9,"visible_child_values":9},
                    {**run.DEFAULT,"slice_rows":0,"visible_child_values":0},
                    {**run.DEFAULT,"child_values":2**31,"visible_child_values":0},
                    {**run.DEFAULT,"child_values":2**31,"visible_child_values":0,"offset_bytes":8}]
        code = render.CALCULATOR.split("const form=")[0]
        script = code+"\nconst profiles="+json.dumps(profiles)+";\n"
        script += "console.log(JSON.stringify(profiles.map(estimateBudget)));\n"
        script += "for(const bad of [{value_bytes:true},{slice_rows:2**31},{parent_bitmap:true},{parent_rows:2**53},{slice_rows:0}])"
        script += "{let rejected=false;try{estimateBudget({...profiles[0],...bad});}catch(e){rejected=true;}"
        script += "if(!rejected)throw Error('invalid accepted');}"
        result = subprocess.run(["node","-e",script],check=True,text=True,capture_output=True)
        self.assertEqual(json.loads(result.stdout),[run.estimate_budget(p) for p in profiles])
        self.assertIn('role="alert"',render.workbench(run.simulate()))


if __name__ == "__main__":
    unittest.main()
