import json
import math
import statistics
import subprocess
import unittest

from experiments.hll_register_union_selection import run as m
from experiments.hll_register_union_selection.render import JS


class HllRegisterTests(unittest.TestCase):
    def test_digest_rank_independent_binary_string_oracle(self):
        digests=list(range(256))+[65535,0,32768,4096]
        self.assertEqual(m.sketch(digests)["registers"],m.independent_register_oracle(digests))
        self.assertEqual(m.register_rank(0),(0,13))
        self.assertEqual(m.register_rank(16),(0,1))
        self.assertEqual(m.register_rank(32),(0,2))

    def test_union_matches_direct_ingestion_with_overlap(self):
        a=list(map(m.original_digest,range(20)));b=list(map(m.original_digest,range(10,30)))
        self.assertEqual(m.merge([m.sketch(a),m.sketch(b)]),m.sketch(a+b))
        self.assertEqual(len(set(range(20))|set(range(10,30))),30)

    def test_register_union_algebra(self):
        a=m.sketch([0,31]);b=m.sketch([128,53]);c=m.sketch([777,19])
        self.assertEqual(m.merge([a,b]),m.merge([b,a]))
        self.assertEqual(m.merge([a,a]),a)
        self.assertEqual(m.merge([m.merge([a,b]),c]),m.merge([a,m.merge([b,c])]))

    def test_duplicates_empty_and_small_range(self):
        state=m.sketch([m.original_digest(0)]*100)
        self.assertEqual(state,m.sketch([m.original_digest(0)]))
        estimate=m.classic_estimate(state)
        self.assertEqual(estimate["selected_mode"],"linear_counting")
        self.assertAlmostEqual(estimate["selected_teaching_estimate"],16*math.log(16/15))
        self.assertEqual(m.classic_estimate(m.sketch([]))["selected_teaching_estimate"],0)

    def test_precision_and_hash_contract_mismatch(self):
        with self.assertRaises(ValueError):
            m.merge([m.sketch([1]),m.sketch([1],5)])
        with self.assertRaises(ValueError):
            m.merge([m.sketch([1]),m.sketch([1],4,17)])
        with self.assertRaises(ValueError):
            m.sketch([-1])

    def test_dense_payload_and_rank_width_gate(self):
        self.assertEqual(m.budget(m.DEFAULT)["selected_dense_register_payload_bytes"],12288)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"register_bits":5})

    def test_node_registers_estimate_and_budget_parity(self):
        source=JS+"\nconsole.log(JSON.stringify({result:simulate("+json.dumps(list(range(20)))+","+json.dumps(list(range(10,30)))+"),budget:budget("+json.dumps(m.DEFAULT)+")}))"
        out=json.loads(subprocess.check_output(["node","-e",source],text=True))
        expected=m.sketch(list(map(m.original_digest,range(30))))
        self.assertEqual(out["result"]["merged_registers"],expected["registers"])
        self.assertAlmostEqual(out["result"]["estimate"]["selected_teaching_estimate"],m.classic_estimate(expected)["selected_teaching_estimate"],places=10)
        self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_timing_paths(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("bitwise","string"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(256);self.assertEqual(a(),b())
        self.assertIsNone(report["native_pfcount_result"])


if __name__=="__main__":
    unittest.main()
