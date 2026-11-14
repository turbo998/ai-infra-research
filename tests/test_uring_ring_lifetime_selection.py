from collections import deque
import json
import random
import statistics
import subprocess
import unittest

from experiments.uring_ring_lifetime_selection import run as m
from experiments.uring_ring_lifetime_selection.render import JS,COMMANDS


class UringRingLifetimeTests(unittest.TestCase):
    def test_unsigned_wrap_and_full_credit(self):
        ring=m.Ring(entries=4,cq_entries=4,buffers=4,start=m.U32-2)
        for i in range(4):
            ring.submit(i,i)
        self.assertEqual((ring.head,ring.tail),(m.U32-2,2))
        self.assertEqual(m.depth(ring.tail,ring.head),4)
        self.assertEqual(ring.snapshot()["sq_free"],0)
        self.assertEqual([ring.consume() for _ in range(4)],list(range(4)))
        self.assertEqual(ring.snapshot()["sq_depth"],0)
        self.assertEqual(len(ring.owned),4)

    def test_sqe_reuse_does_not_release_data_buffer(self):
        ring=m.Ring();ring.submit(100,0);ring.submit(101,1);ring.consume()
        self.assertEqual(ring.submit(102,2),0)
        self.assertEqual(ring.sq[0][0],102)
        self.assertEqual(ring.owned[0],100)
        with self.assertRaises(ValueError):
            ring.submit(103,0)
        ring.complete(100,4);result=ring.reap()
        self.assertEqual(result["buffer"],0)
        self.assertEqual(ring.owned[2],102)

    def test_independent_deque_and_identity_oracle(self):
        rng=random.Random(140);ring=m.Ring(entries=4,cq_entries=4,buffers=8,start=m.U32-3)
        queued=deque();active={};done=deque();next_id=0
        for _ in range(160):
            free=[v for v in range(8) if v not in active.values()]
            actions=[]
            if free and len(queued)<4:
                actions.append("submit")
            if queued:
                actions.append("consume")
            inflight=[i for i in active if i not in queued and i not in done]
            if inflight and len(done)<4:
                actions.append("complete")
            if done:
                actions.append("reap")
            action=rng.choice(actions)
            if action=="submit":
                buffer=free[0];ring.submit(next_id,buffer);queued.append(next_id);active[next_id]=buffer;next_id+=1
            elif action=="consume":
                self.assertEqual(ring.consume(),queued.popleft())
            elif action=="complete":
                request=rng.choice(inflight);ring.complete(request,4);done.append(request)
            else:
                request=done.popleft();self.assertEqual(ring.reap()["buffer"],active.pop(request))
            self.assertEqual(ring.snapshot()["sq_depth"],len(queued))
            self.assertEqual(ring.snapshot()["cq_depth"],len(done))
            self.assertEqual(ring.owned,{v:k for k,v in active.items()})

    def test_negative_short_and_complete_results_are_distinct(self):
        ring=m.Ring(entries=4,cq_entries=4,buffers=3)
        for i in range(3):
            ring.submit(i,i);ring.consume()
        for i,result in enumerate((-5,2,4)):
            ring.complete(i,result)
        self.assertEqual([ring.reap()["status"] for _ in range(3)],["error","short_or_eof","complete"])

    def test_bad_ids_capacity_and_completion_state_fail(self):
        with self.assertRaises(ValueError):
            m.Ring(entries=3)
        ring=m.Ring();ring.submit(1,0)
        with self.assertRaises(ValueError):
            ring.complete(1,4)
        ring.consume()
        with self.assertRaises(ValueError):
            ring.complete(True,4)
        ring.complete(1,4);ring.reap()
        with self.assertRaises(ValueError):
            ring.submit(1,0)
        with self.assertRaises(ValueError):
            ring.complete(1,4)
        ring=m.Ring(entries=2,cq_entries=1,buffers=2)
        ring.submit(1,0);ring.consume();ring.submit(2,1);ring.consume();ring.complete(1,4)
        with self.assertRaises(ValueError):
            ring.complete(2,4)
        ring.reap();ring.complete(2,4)
        self.assertEqual(ring.reap()["request_id"],2)

    def test_selected_entry_and_pool_payload(self):
        out=m.budget(m.DEFAULT)
        self.assertEqual(out["selected_ring_entry_payload_bytes"],768)
        self.assertEqual(out["selected_buffer_pool_payload_bytes"],33554432)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"cq_entries":3})

    def test_node_python_state_and_budget_parity(self):
        p={"entries":2,"cq_entries":4,"buffers":3,"start":0}
        out=json.loads(subprocess.check_output(["node","-e",JS+"\nconsole.log(JSON.stringify({trace:play("+json.dumps(p)+","+json.dumps(COMMANDS)+"),budget:budget("+json.dumps(m.DEFAULT)+")}))"],text=True))
        ring=m.Ring()
        for command,actual in zip(COMMANDS,out["trace"]):
            op,*args=command;getattr(ring,op)(*args)
            self.assertEqual(actual["snapshot"],ring.snapshot())
        self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_timing_records(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("cursor","shifting"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(256);self.assertEqual(a(),b())
        self.assertIsNone(report["native_iops"])


if __name__=="__main__":
    unittest.main()
