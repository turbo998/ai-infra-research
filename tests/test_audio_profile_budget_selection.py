import copy
import json
import random
import subprocess
import unittest
from experiments.audio_profile_budget_selection.run import DEFAULT,plan,tokens
from experiments.audio_profile_budget_selection.render import JS


class AudioProfileTests(unittest.TestCase):
    def test_default_budget_and_legacy_shortfall(self):
        a=plan(DEFAULT);b=plan(DEFAULT,"legacy_fixed")
        self.assertEqual((a["selected_samples"],a["declared_raw_mel_frames_per_item"],a["total_audio_tokens"]),(480000,3000,375))
        self.assertEqual((b["total_audio_tokens"],b["budget_shortfall_per_item"]),(13,362))

    def test_sample_rate_not_fixed_waveform_duration(self):
        p={**DEFAULT,"mel_sr":48000,"hop":480}
        self.assertEqual(plan(p)["selected_samples"],1440000)
        self.assertEqual(plan(p)["total_audio_tokens"],375)
        self.assertNotEqual(plan(p,"legacy_fixed")["total_audio_tokens"],375)

    def test_override_clamp_and_zero_truthiness(self):
        self.assertEqual(plan({**DEFAULT,"override_samples":0}),plan(DEFAULT))
        a=plan({**DEFAULT,"override_samples":16000})
        self.assertEqual(a["budget_shortfall_per_item"],362)
        b=plan({**DEFAULT,"override_samples":960000})
        self.assertEqual(b["selected_samples"],480000)
        self.assertEqual(b["warnings"],["override_above_selected_max_ignored"])

    def test_markers_are_not_audio_tokens(self):
        a=plan({**DEFAULT,"time_markers":True})
        self.assertEqual((a["total_audio_tokens"],a["marker_digit_tokens_per_item"],a["audio_span_tokens"]),(375,26,403))
        self.assertEqual(plan(DEFAULT)["audio_span_tokens"],377)

    def test_context_does_not_truncate_budget(self):
        a=plan({**DEFAULT,"seq_len":128})
        self.assertEqual(a["declared_audio_budget"],{"audio":375})
        self.assertFalse(a["context_ledger_fits"])

    def test_shared_waveform_dense_batch_and_no_audio(self):
        a=plan(DEFAULT);b=plan({**DEFAULT,"audio_count":3})
        self.assertEqual(a["declared_unique_numpy_waveform_bytes"],b["declared_unique_numpy_waveform_bytes"])
        self.assertEqual(b["declared_float32_mel_batch_bytes"],3*a["declared_float32_mel_batch_bytes"])
        empty=plan({**DEFAULT,"audio_count":0})
        self.assertEqual(empty["declared_audio_budget"],{})
        self.assertEqual(empty["declared_unique_numpy_waveform_bytes"],0)
        self.assertIsNone(empty["native_peak_bytes"])

    def test_integer_ceil_equivalence_and_invalid_inputs(self):
        rng=random.Random(169)
        for x in [0,1,7,8,9,3000]+[rng.randrange(1000001) for _ in range(64)]:
            self.assertEqual(tokens(x),tokens(x,False))
        for change in ({"hop":0},{"audio_count":True},{"override_samples":-1},{"time_markers":1}):
            with self.subTest(change=change),self.assertRaises(ValueError):plan({**DEFAULT,**change})

    def test_node_parity65profiles(self):
        rng=random.Random(169);profiles=[copy.deepcopy(DEFAULT)]
        for _ in range(64):
            profiles.append({**DEFAULT,"mel_sr":rng.choice([8000,16000,22050,24000,44100,48000]),
                             "hop":rng.randrange(1,1025),"mel_dim":rng.randrange(1,257),"audio_count":rng.randrange(5),
                             "override_samples":rng.choice([None,0,16000,100000,2000000]),
                             "seq_len":rng.randrange(1,100001),"text_tokens":rng.randrange(5000),
                             "time_markers":rng.choice([True,False])})
        expected=[[plan(p,s) for s in ("rate_aligned","legacy_fixed")] for p in profiles]
        script=JS+"\nconst ps="+json.dumps(profiles)+";console.log(JSON.stringify(ps.map(p=>['rate_aligned','legacy_fixed'].map(s=>plan(p,s)))));"
        out=subprocess.run(["node"],input=script,check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(out.stdout),expected)
