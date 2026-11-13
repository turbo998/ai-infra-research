import json
import re
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.vfr_timestamp_sampling_selection import render, run


class VfrTimestampTests(unittest.TestCase):
    def test_binary_matches_independent_half_open_interval_oracle(self):
        timeline = run.Timeline((1000,1040,1080,1200,1240),1400)
        requests = list(range(1000,1400))
        self.assertEqual(run.binary_lookup(timeline,requests),run.interval_oracle(timeline,requests))
        self.assertEqual(run.binary_lookup(timeline,[1039,1040,1079,1080,1399]),[0,1,1,2,4])
        for value in (999,1400,True,1040.0):
            with self.assertRaises(ValueError):
                run.binary_lookup(timeline,[value])
        for pts,end in (((1,1),2),((2,1),3),((1,2),2),([],2),([1,2],3)):
            with self.assertRaises(ValueError):
                run.Timeline(pts,end)

    def test_vfr_approximation_cfr_control_and_stream_origin(self):
        s = run.simulate()["sampling"]
        self.assertEqual(s["approximate_mismatches"],5)
        self.assertEqual(s["denominator"],9)
        self.assertEqual((s["cfr_requests"],s["cfr_mismatches"]),(200,0))
        original = run.Timeline((1000,1040,1080,1200,1240),1400)
        shifted = run.Timeline(tuple(t-2000 for t in original.pts),-600)
        requests = [1000,1040,1199,1399]
        self.assertEqual(run.binary_lookup(original,requests),
                         run.binary_lookup(shifted,[t-2000 for t in requests]))
        self.assertEqual(run.average_fps_lookup(original,requests,25,2),
                         run.average_fps_lookup(shifted,[t-2000 for t in requests],25,2))
        with self.assertRaises(ValueError):
            run.average_fps_lookup(original,[1399],1000,1)
        self.assertTrue(all(index>=5 for index in s["wrong_ignore_origin_raw_indices"]))

    def test_duplicate_targets_restore_order_without_losing_slots(self):
        d = run.simulate()["duplicates"]
        self.assertEqual(d["unique_targets"],[1,2])
        self.assertEqual(d["output_to_unique"],[1,1,0,1])
        self.assertEqual(d["restored_slots"],[2,2,1,2])
        self.assertEqual((d["output_slot_count"],d["unique_target_count"]),(4,2))
        self.assertIsNone(d["actual_decoded_frame_count"])
        self.assertEqual(run.restore_slots([])["restored_slots"],[])
        with self.assertRaises(ValueError):
            run.restore_slots([True])

    def test_tail_padding_wrap_is_valid_list_not_whole_video_modulo(self):
        r = run.simulate()
        p = r["padding"]
        self.assertEqual(p["wrap"]["frame_indices"],[2,4,4,2,4])
        self.assertEqual(p["wrong_whole_stream_modulo_indices"],[2,4,4,0,2])
        self.assertEqual(p["repeat_last"]["frame_indices"],[2,4,4,4,4])
        self.assertEqual(p["repeat_last_requested_not_last_video_frame"],[2,2,2,2])
        timeline = run.Timeline((1000,1040,1080,1200,1240),1400)
        with self.assertRaisesRegex(ValueError,"half-open"):
            run.clip_timestamps(timeline,1160,5,80,"error")
        for count,step,policy in ((0,1,"wrap"),(2,0,"wrap"),(2,1,"clamp")):
            with self.assertRaises(ValueError):
                run.clip_timestamps(timeline,1000,count,step,policy)

    def test_unsorted_records_and_varied_timeline_parity(self):
        r = run.simulate()
        self.assertEqual(r["ordering"]["presentation_labels"],list("ABCDE"))
        self.assertNotEqual(r["ordering"]["unsorted_records"],r["ordering"]["presentation_records"])
        self.assertTrue(all(row["index_mismatches"]==0 for row in r["variants"]))
        for count in (1,2,9):
            timeline = run.fixture_timeline(count)
            qs = list(timeline.pts)+[timeline.end-1]
            self.assertEqual(run.binary_lookup(timeline,qs),run.interval_oracle(timeline,qs))

    def test_exact_output_budget_preserves_duplicate_slots(self):
        b = run.estimate_budget(run.DEFAULT)
        fewer = run.estimate_budget({**run.DEFAULT,"unique_targets":1})
        self.assertEqual(b["output_tensor_bytes"],77070336)
        self.assertEqual(b["pts_and_duration_float64_bytes"],8192)
        self.assertEqual(b["output_tensor_bytes"],fewer["output_tensor_bytes"])
        self.assertLess(fewer["unique_target_tensor_payload_bytes"],b["unique_target_tensor_payload_bytes"])
        self.assertIsNone(b["actual_decoded_frame_count"])
        for key,value in (("unique_targets",0),("unique_targets",513),("scalar_bytes",True),
                          ("height",2**53),("clips",True)):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,key:value})

    def test_sealed_artifacts_csp_svg_and_raw_timing_samples(self):
        for path,content in run.outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
        r = run.simulate()
        for row in r["latency"]["rows"]:
            self.assertEqual(len(row["primary_samples_ms"]),7)
            self.assertEqual(row["primary_median_ms"],statistics.median(row["primary_samples_ms"]))
            self.assertEqual(row["baseline_median_ms"],statistics.median(row["baseline_samples_ms"]))
            self.assertEqual(row["index_mismatches"],0)
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

    def test_javascript_calculator_parity_and_invalid_inputs(self):
        profiles = [run.DEFAULT,{**run.DEFAULT,"unique_targets":1},
                    {**run.DEFAULT,"scalar_bytes":4},
                    {**run.DEFAULT,"clips":1,"frames_per_clip":1,"unique_targets":1},
                    {**run.DEFAULT,"height":1080,"width":1920,"scalar_bytes":2}]
        code = render.CALCULATOR.split("const form=")[0]
        script = code+"\nconst profiles="+json.dumps(profiles)+";\n"
        script += "console.log(JSON.stringify(profiles.map(estimateBudget)));\n"
        script += "for(const bad of [{unique_targets:0},{unique_targets:513},{scalar_bytes:true},{height:2**53},{clips:true}])"
        script += "{let rejected=false;try{estimateBudget({...profiles[0],...bad});}catch(e){rejected=true;}"
        script += "if(!rejected)throw Error('invalid accepted');}"
        result = subprocess.run(["node","-e",script],check=True,text=True,capture_output=True)
        self.assertEqual(json.loads(result.stdout),[run.estimate_budget(p) for p in profiles])
        self.assertIn('role="alert"',render.workbench(run.simulate()))


if __name__ == "__main__":
    unittest.main()
