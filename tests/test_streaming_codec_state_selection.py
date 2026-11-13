from dataclasses import replace
import json
import math
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.streaming_codec_state_selection.run import (
    AlignedIngress, Budget, CausalConv, FRAME_SIGNAL, SIGNAL, TransposeConv, WEIGHTS,
    chunks, estimate, offline_causal, offline_transpose, outputs, simulate, stream_causal,
)


class StreamingCodecStateTests(unittest.TestCase):
    def test_irregular_packets_match_independent_offline_formula(self):
        reference = offline_causal(SIGNAL, WEIGHTS, 4, 0.125)
        for pattern in ([17], [1]*17, [1,2,5,3,6], [4,4,4,5]):
            actual, padding = stream_causal(pattern)
            self.assertEqual(len(actual), 5)
            self.assertEqual(padding, 3)
            for value, expected in zip(actual, reference):
                self.assertAlmostEqual(value, expected, places=12)

    def test_transpose_overlap_handles_bias_and_no_overlap(self):
        for weights, stride in (([0.2], 1), (WEIGHTS, 1), (WEIGHTS, 4), ([0.2,0.3], 2)):
            conv = TransposeConv(weights, stride, 0.4)
            actual = []
            for frame in FRAME_SIGNAL:
                actual.extend(conv.step([frame]))
            expected = offline_transpose(FRAME_SIGNAL, weights, stride, 0.4)
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assertAlmostEqual(value, reference, places=12)

    def test_dilation_and_interleaved_streams_keep_independent_history(self):
        other = list(reversed(SIGNAL))
        streams = [AlignedIngress(CausalConv(WEIGHTS, 4, 0.125, 2)) for _ in range(2)]
        actual = [[], []]
        for i in range(len(SIGNAL)):
            for index, signal in enumerate((SIGNAL, other)):
                actual[index].extend(streams[index].feed([signal[i]]))
        for index, signal in enumerate((SIGNAL, other)):
            actual[index].extend(streams[index].finish())
            expected = offline_causal(signal, WEIGHTS, 4, 0.125, 2)
            for value, reference in zip(actual[index], expected):
                self.assertAlmostEqual(value, reference, places=12)

    def test_eos_rejection_and_closed_stream_errors_are_explicit(self):
        ingress = AlignedIngress(CausalConv(WEIGHTS, 4))
        self.assertEqual(ingress.feed([1.0]), [])
        with self.assertRaisesRegex(ValueError, "incomplete stride"):
            ingress.finish("reject")
        ingress.feed([0.0]*3)
        self.assertEqual(ingress.finish("reject"), [])
        with self.assertRaisesRegex(ValueError, "closed"):
            ingress.feed([1.0])
        with self.assertRaisesRegex(ValueError, "already finished"):
            ingress.finish()
        ingress.reset()
        ingress.feed([9.0])
        ingress.reset()
        self.assertEqual(ingress.feed(SIGNAL[:4]), CausalConv(WEIGHTS, 4).step(SIGNAL[:4]))
        with self.assertRaises(ValueError):
            CausalConv(WEIGHTS, 4).step([1.0])
        with self.assertRaises(ValueError):
            CausalConv(WEIGHTS, 4).step([math.nan]*4)

    def test_negative_controls_and_fresh_slot_reset(self):
        cases = {row["id"]: row for row in simulate()["cases"]}
        for name in ("aligned-ingress-and-EOS-padding", "transpose-bias-free-carry", "reset-slot-before-new-request"):
            self.assertTrue(cases[name]["matches_reference"])
        for name in ("reset-history-every-chunk", "transpose-carry-includes-bias", "reuse-slot-with-stale-history"):
            self.assertFalse(cases[name]["matches_reference"])
            self.assertGreater(cases[name]["maximum_absolute_error"], 0.3)
        dropped = cases["drop-transport-remainders"]
        self.assertFalse(dropped["same_length"])
        self.assertIsNone(dropped["maximum_absolute_error"])
        conv = TransposeConv(WEIGHTS, 4, 0.4)
        conv.step([0.5])
        conv.reset()
        self.assertEqual(conv.partial, [0.0]*3)

    def test_state_budget_counts_distinct_buffers_and_validates_alignment(self):
        result = estimate(Budget())
        self.assertEqual(result["conv_history_bytes"], 12288)
        self.assertEqual(result["total_state_bound_bytes"], 3*12288)
        self.assertEqual(result["chunk_collection_bound_ms"], 40)
        dilated = estimate(replace(Budget(), dilation=2))
        self.assertEqual(dilated["conv_history_bytes"], 3*12288)
        self.assertEqual(dilated["transpose_partial_bytes"], 12288)
        for changes in ({"chunk_samples": 961}, {"kernel": 3}, {"scalar_bytes": 3}, {"stage_rate_hz": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                estimate(replace(Budget(), **changes))

    def test_sealed_artifacts_csp_seven_cards_and_geometry(self):
        for path, content in outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"), content)
            if path.name.endswith("-cards.html"):
                self.assertEqual(content.count('<section class="card"'), 7)
                self.assertNotIn("<script", content)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'", content)
            if path.suffix == ".svg":
                root = ET.fromstring(content)
                self.assertEqual(root.attrib["viewBox"], "0 0 1600 1000")
                for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
                    for point in polygon.attrib["points"].split():
                        self.assertTrue(200 <= float(point.split(",")[1]) <= 620)

    @unittest.skipUnless(shutil.which("node"), "Node.js optional")
    def test_javascript_budget_parity_and_invalid_input(self):
        from experiments.streaming_codec_state_selection.render import CALCULATOR
        source = CALCULATOR.split("const form=")[0]
        profiles = [Budget(), replace(Budget(), dilation=2), replace(Budget(), kernel=4, stride=4)]
        script = source + "\nconsole.log(JSON.stringify(" + json.dumps([estimate(p)["input"] for p in profiles]) + ".map(estimateBudget)));"
        actual = json.loads(subprocess.run(["node","-e",script], check=True, capture_output=True, text=True).stdout)
        for profile,result in zip(profiles,actual):
            for key,value in result.items():
                self.assertAlmostEqual(value, estimate(profile)[key], places=9)
        invalid = source + "\ntry{estimateBudget(" + json.dumps({**estimate(Budget())["input"],"chunk_samples":961}) + ");process.exit(1)}catch(e){console.log(e.message)}"
        result = subprocess.run(["node","-e",invalid], check=True, capture_output=True, text=True)
        self.assertIn("align", result.stdout)


if __name__ == "__main__":
    unittest.main()
