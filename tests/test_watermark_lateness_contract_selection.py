import json
import statistics
import subprocess
import unittest

from experiments.watermark_lateness_contract_selection import run as m
from experiments.watermark_lateness_contract_selection.render import JS


class WatermarkLatenessTests(unittest.TestCase):
    def test_bounded_disorder_and_unknown(self):
        self.assertIsNone(m.bounded_watermark(None,2))
        self.assertEqual(m.bounded_watermark(29,2),26)
        self.assertEqual(m.bounded_watermark(0,0),-1)
        v=m.Valve(2)
        self.assertIsNone(v.watermark(0,26))
        self.assertEqual(v.watermark(1,6),6)

    def test_idle_resume_alignment_and_monotonicity(self):
        v=m.Valve(2);v.watermark(0,26);v.watermark(1,6)
        self.assertEqual(v.idle(1),26)
        self.assertEqual(v.resume(1),26)
        self.assertFalse(v.aligned[1])
        self.assertEqual(v.watermark(0,36),36)
        self.assertEqual(v.watermark(1,35),36)
        self.assertFalse(v.aligned[1])
        self.assertEqual(v.watermark(1,40),36)
        self.assertTrue(v.aligned[1])
        self.assertEqual(v.watermark(0,39),39)
        self.assertEqual(v.watermark(0,20),39)

    def test_all_idle_flush_order_independence(self):
        finals=[]
        for order in ((0,1),(1,0)):
            v=m.Valve(2);v.watermark(0,10);v.watermark(1,20)
            for p in order:
                v.idle(p)
            finals.append(v.output)
            self.assertEqual(v.watermark(0,100),20)
        self.assertEqual(finals,[20,20])

    def test_window_fire_late_update_and_cleanup(self):
        w=m.Windows(10,5);w.event(3);w.event(8)
        self.assertEqual(w.advance(8),[])
        self.assertEqual(w.advance(9),[{"kind":"fire","window_start":0,"count":2}])
        self.assertEqual(w.event(8)["status"],"accepted_late_update")
        w.advance(13)
        self.assertEqual(w.event(1)["status"],"accepted_late_update")
        self.assertEqual(w.advance(14),[{"kind":"cleanup","window_start":0}])
        self.assertEqual(w.event(8)["status"],"rejected_window_cleaned")
        self.assertEqual(w.counts,{})

    def test_discrete_window_membership_oracle(self):
        for size in (1,3,10):
            w=m.Windows(size,5)
            for t in range(-12,13):
                start,fire,cleanup=w.times(t)
                self.assertIn(t,range(start,start+size))
                self.assertEqual(fire,max(range(start,start+size)))
                self.assertEqual(cleanup,fire+5)
        w=m.Windows(10,0);w.event(1)
        self.assertEqual([v["kind"] for v in w.advance(9)],["fire","cleanup"])
        self.assertEqual(w.event(1)["status"],"rejected_window_cleaned")

    def test_heap_oracle_and_invalid_controls(self):
        for n in (1,2,7,64):
            self.assertEqual(m.minimum_trace(n,True),m.minimum_trace(n,False))
        with self.assertRaises(ValueError):
            m.bounded_watermark(5,-1)
        with self.assertRaises(ValueError):
            m.Valve(1).idle(1)
        w=m.Windows(10,5);w.advance(9)
        with self.assertRaises(ValueError):
            w.advance(8)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"active_keys":2**53})

    def test_node_window_and_budget_parity(self):
        for watermark in (8,9,13,14):
            source=JS+f"\nconsole.log(JSON.stringify({{result:simulate(29,9,2,10,5,{watermark},8),budget:budget("+json.dumps(m.DEFAULT)+")}))"
            out=json.loads(subprocess.check_output(["node","-e",source],text=True))
            w=m.Windows(10,5);w.advance(watermark)
            self.assertEqual(out["result"]["event_status"],w.event(8)["status"])
            self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_timing_medians(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("heap","scan"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        self.assertFalse(report["exactly_once_reproduced"])


if __name__=="__main__":
    unittest.main()
