import copy
import json
import random
import subprocess
import unittest

from experiments.image_mode_channel_selection import run as m
from experiments.image_mode_channel_selection.render import JS,cards


class ImageModeTests(unittest.TestCase):
    def test_independent_source_mode_literals(self):
        expected={"1":[0,0,0,255,255,255],"L":[18,18,18,207,207,207],"LA":[18,18,18,207,207,207],
                  "P":[9,70,220,190,25,14],"RGB":[9,70,220,190,25,14],"RGBA":[9,70,220,190,25,14]}
        for row in m.fixtures():
            p=row["inputs"];r=row["outputs"];v=expected[p["mode"]]
            target=v if p["pixel_format"]=="RGB" else [v[2],v[1],v[0],v[5],v[4],v[3]]
            self.assertEqual(r["pixels"],target);self.assertEqual(r["shape"],[1,2,3]);self.assertEqual(r["dtype"],"uint8")

    def test_alpha_discard_not_composite(self):
        p=copy.deepcopy(m.DEFAULT);snapshot=copy.deepcopy(p);a=m.evaluate(p)
        p["pixels"][3]=255;p["pixels"][7]=0;b=m.evaluate(p)
        self.assertEqual(a["pixels"],b["pixels"]);self.assertEqual(a["pixels"][:3],[220,70,9])
        m.evaluate(snapshot);self.assertEqual(snapshot,m.DEFAULT)

    def test_unsafe_axes_and_channel_counts(self):
        rows=m.fixtures()
        self.assertEqual(sum(not r["outputs"]["naive_matches_contract"] for r in rows),10)
        for row in rows:
            if row["inputs"]["mode"] in ("1","L","P") and row["inputs"]["pixel_format"]=="BGR":
                self.assertIsNone(row["outputs"]["naive_pixels"]);self.assertIsNotNone(row["outputs"]["naive_error"])
        self.assertEqual(m.evaluate(m.DEFAULT)["naive_pixels"][:4],[0,220,70,9])

    def test_consumer_label_and_double_swap(self):
        r=m.evaluate(m.DEFAULT)
        self.assertEqual((r["toy_rgb_score"],r["toy_score_if_output_mislabeled_rgb"]),(259,364))
        self.assertEqual(m.reverse_channels(m.reverse_channels(r["canonical_rgb"])),r["canonical_rgb"])

    def test_invalid_palette_shape_type_and_target(self):
        changes=[{"width":0},{"height":True},{"mode":"CMYK"},{"pixel_format":"Grayscale"},
                 {"pixels":[9,70,220]},{"palette":[]},{"pixels":[-1,70,220,0,190,25,14,255]},
                 {"pixels":[True,70,220,0,190,25,14,255]}]
        for change in changes:
            p={**m.DEFAULT,**change}
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.evaluate(p)
        p={"mode":"P","height":1,"width":1,"pixels":[1],"palette":[[1,2,3]],"pixel_format":"RGB"}
        with self.assertRaises(ValueError):
            m.evaluate(p)

    def test_independent_permutation_and_cpu_paths(self):
        rng=random.Random(150)
        for n in (1,2,7,64):
            data=[rng.randrange(256) for _ in range(n*3)]
            expected=[data[i+(2-j)] for i in range(0,len(data),3) for j in range(3)]
            self.assertEqual(m.reverse_channels(data),expected)
            self.assertEqual(m.reverse_channels(expected),data)
            funcs=m.timing_paths(n);self.assertEqual(funcs[0](),funcs[1]())

    def test_node_parity_for_modes_and_random_colours(self):
        rng=random.Random(151);inputs=[r["inputs"] for r in m.fixtures()]
        for i in range(32):
            n=i%8+1;values=[rng.randrange(256) for _ in range(n*4)]
            inputs.append({"mode":"RGBA","height":1,"width":n,"pixels":values,"palette":None,"pixel_format":"BGR" if i%2 else "RGB"})
        result=subprocess.run(["node"],input=JS+"\nconsole.log(JSON.stringify("+json.dumps(inputs)+".map(evaluate)));",
                              capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[m.evaluate(p) for p in inputs])

    def test_sealed_report_full_twelve_case_matrix(self):
        latency=json.loads((m.HERE/"latency.json").read_text())
        sealed=json.loads((m.HERE/"report.json").read_text())
        self.assertEqual(sealed,m.build_report(latency));self.assertFalse(sealed["independently_reproduced"])
        self.assertIsNone(sealed["native_bgr_contiguous"]);self.assertFalse(sealed["robustness"]["alpha_changes_output"])
        html=cards(sealed);self.assertEqual(html.count('class="card"'),7)
        self.assertEqual(html.count("✕ Normalize first"),10)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",html);self.assertNotIn("<script",html)
