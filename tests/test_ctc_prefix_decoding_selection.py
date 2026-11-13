from dataclasses import replace
import itertools
import json
import math
import shutil
import statistics
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.ctc_prefix_decoding_selection.run import (
    Budget, FRAMES, NEG_INF, REPEATS, STREAM, collapse, concat_chunk_distributions,
    decode_chunks, difference, enumerate_paths, estimate, load_latency, outputs,
    prefix_decode, probabilities, simulate, target_log_probability,
)


class CtcPrefixDecodingTests(unittest.TestCase):
    def test_exact_prefix_and_independent_target_forward_match_path_enumeration(self):
        profiles = [[],FRAMES,REPEATS,STREAM,[[1,0,0],[0,1,0],[0,0,1]],
                    [[0,0.5,0.5]]*4,[[0.2,0.3,0.1,0.4]]*4]
        for frames in profiles:
            reference = enumerate_paths(frames)
            self.assertLess(difference(probabilities(prefix_decode(frames)),reference),1e-12)
            self.assertAlmostEqual(math.fsum(reference.values()),1)
            for target,probability in reference.items():
                actual = target_log_probability(frames,target)
                self.assertAlmostEqual(0 if actual == NEG_INF else math.exp(actual),probability,places=12)
        self.assertEqual(target_log_probability(STREAM,(1,1)),NEG_INF)

    def test_greedy_path_and_summed_transcript_answer_different_questions(self):
        report = simulate()
        self.assertEqual(report["greedy"]["transcript"],"")
        self.assertEqual(report["greedy"]["path_probability"],0.2025)
        self.assertEqual(report["best_transcript"],{"text":"a","probability":0.4375})
        viterbi = enumerate_paths(FRAMES,use_max=True)
        self.assertEqual(max(viterbi,key=viterbi.get),())
        self.assertAlmostEqual(viterbi[(1,)],0.1575)
        cases = {row["id"]:row for row in report["cases"]}
        self.assertTrue(cases["best-path-per-transcript"]["intentional_search_policy"])
        self.assertTrue(cases["beam-one-pruning"]["intentional_search_policy"])
        self.assertLess(cases["beam-one-pruning"]["total_score_mass"],1)

    def test_blank_separates_repeated_labels_and_must_not_be_removed_first(self):
        self.assertEqual(collapse((1,1,0,1,2,2)),(1,1,2))
        self.assertEqual(collapse((1,1,0,1,2,2),True),(1,2))
        self.assertAlmostEqual(enumerate_paths(REPEATS)[(1,1)],0.384)
        self.assertNotIn((1,1),probabilities(prefix_decode(STREAM)))
        broken = probabilities(prefix_decode(STREAM,merge_repeat_ends=True))
        self.assertAlmostEqual(broken[(1,1)],0.64)
        self.assertGreater(math.fsum(broken.values()),1)

    def test_chunk_carry_preserves_every_cut_and_reset_invents_impossible_repeat(self):
        reference = probabilities(prefix_decode(REPEATS))
        for cuts in itertools.product((False,True),repeat=2):
            boundaries = [0]+[i+1 for i,cut in enumerate(cuts) if cut]+[3]
            sizes = [end-start for start,end in zip(boundaries,boundaries[1:])]
            self.assertLess(difference(decode_chunks(REPEATS,sizes),reference),1e-12)
        state = prefix_decode(STREAM[:1])
        before = dict(state)
        self.assertEqual(prefix_decode([],state),state)
        self.assertLess(difference(probabilities(prefix_decode(STREAM[1:],state)),
                                   enumerate_paths(STREAM)),1e-12)
        self.assertEqual(state,before)
        reset = concat_chunk_distributions([STREAM[:1],STREAM[1:]])
        self.assertAlmostEqual(reset[(1,1)],0.64)
        with self.assertRaises(ValueError):
            decode_chunks(REPEATS,[1,1])

    def test_log_space_retains_positive_underflowed_event_and_rejects_bad_inputs(self):
        underflow = simulate()["underflow"]
        self.assertEqual(underflow["linear_probability"],0)
        self.assertTrue(math.isfinite(underflow["log_probability"]))
        self.assertAlmostEqual(underflow["log_probability"],underflow["closed_form_log_probability"],places=8)
        for frames in ([[0.2,0.2]],[[1,0],[0,1,0]],[[1.1,-0.1]],[[math.nan,0]],
                       [[math.inf,0]],[[1]]):
            with self.assertRaises(ValueError):
                prefix_decode(frames)
        for width in (0,-1,True,1.5):
            with self.assertRaises(ValueError):
                prefix_decode(FRAMES,beam_width=width)
        with self.assertRaises(ValueError):
            target_log_probability([], (0,))
        with self.assertRaises(ValueError):
            prefix_decode(FRAMES,{})

    def test_logical_budget_counts_and_numerical_or_index_limits(self):
        result = estimate(Budget())
        self.assertIsNone(result["path_count_if_js_safe"])
        self.assertEqual(result["per_frame_contribution_bound"],2176)
        self.assertEqual(result["two_log_scores_bytes"],1024)
        self.assertEqual(result["prefix_token_payload_upper_bytes"],25600)
        self.assertEqual(result["audio_span_ms"],2000)
        self.assertEqual(estimate(replace(Budget(),frames=9,labels=2))["path_count_if_js_safe"],19683)
        for profile in (replace(Budget(),frames=0),replace(Budget(),labels=True),
                        replace(Budget(),token_index_bytes=3),replace(Budget(),frame_ms=math.nan),
                        replace(Budget(),labels=65536,token_index_bytes=2),
                        replace(Budget(),beam_width=9007199254740991),
                        replace(Budget(),frame_ms=1e308)):
            with self.assertRaises(ValueError):
                estimate(profile)

    def test_sealed_samples_reproducibility_csp_and_solid_geometry(self):
        latency = load_latency()
        for row in latency["rows"]:
            for key in ("prefix_dp","path_enumeration"):
                self.assertEqual(row[key+"_median_ms"],statistics.median(row["samples_ns"][key])/1e6)
        for path,content in outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
            if path.name.endswith("-cards.html"):
                self.assertEqual(content.count('<section class="card"'),7)
                self.assertNotIn("<script",content)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'",content)
                self.assertIn("0.4375",content)
            if path.suffix == ".svg":
                root = ET.fromstring(content)
                self.assertEqual(root.attrib["viewBox"],"0 0 1600 1000")
                for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
                    self.assertTrue(all(200 <= float(point.split(",")[1]) <= 620
                                        for point in polygon.attrib["points"].split()))

    @unittest.skipUnless(shutil.which("node"),"Node.js optional")
    def test_javascript_exact_path_bound_payload_parity_and_invalid_input(self):
        from experiments.ctc_prefix_decoding_selection.render import CALCULATOR
        profiles = [Budget(),replace(Budget(),frames=9,labels=2),
                    replace(Budget(),frames=1,labels=65535,token_index_bytes=2),
                    replace(Budget(),labels=3,frames=27,beam_width=1)]
        source = CALCULATOR.split("const form=")[0]
        script = source+"\nconsole.log(JSON.stringify("+json.dumps([estimate(p)["input"] for p in profiles])+".map(estimateBudget)));"
        actual = json.loads(subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)
        for profile,result in zip(profiles,actual):
            for key,value in result.items():
                if value is None or isinstance(value,int):
                    self.assertEqual(value,estimate(profile)[key])
                else:
                    self.assertAlmostEqual(value,estimate(profile)[key],places=9)
        invalid = {**estimate(Budget())["input"],"labels":65536,"token_index_bytes":2}
        script = source+"\ntry{estimateBudget("+json.dumps(invalid)+");process.exit(1)}catch(e){console.log(e.message)}"
        self.assertIn("index",subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)


if __name__ == "__main__":
    unittest.main()
