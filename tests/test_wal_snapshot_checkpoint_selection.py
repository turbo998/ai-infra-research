import json
import random
import subprocess
import unittest

from experiments.wal_snapshot_checkpoint_selection import run as m
from experiments.wal_snapshot_checkpoint_selection.render import JS


class WalSnapshotTests(unittest.TestCase):
    def test_independent_transaction_snapshot_oracle(self):
        rng=random.Random(142);model=m.Wal({1:"A0",2:"B0"});versions=[{1:"A0",2:"B0"}];readers={}
        for i in range(40):
            if i%4==0:
                name=f"r{i}";model.begin(name);readers[name]=dict(versions[-1])
            data={rng.randint(1,2):f"v{i}"};model.commit(data)
            latest={**versions[-1],**data};versions.append(latest)
            for name,expected in readers.items():
                self.assertEqual({page:model.read(page,name) for page in expected},expected)
            self.assertEqual({page:model.read(page) for page in latest},latest)
            model.checkpoint()

    def test_checkpoint_boundary_advances_when_old_readers_end(self):
        w=m.Wal({1:"A0",2:"B0"});w.begin("old");w.commit({1:"A1"});w.begin("middle");w.commit({1:"A2",2:"B2"})
        self.assertEqual(w.checkpoint()["backfilled"],0)
        w.release("old");self.assertEqual(w.checkpoint()["backfilled"],1)
        self.assertEqual(w.read(1,"middle"),"A1")
        w.release("middle");self.assertTrue(w.checkpoint()["fully_backfilled"])
        self.assertEqual(w.db,{1:"A2",2:"B2"})
        self.assertEqual(len(w.frames),3)

    def test_committed_end_mark_hides_uncommitted_tail(self):
        w=m.Wal({1:"A0",2:"B0"});w.commit({1:"A1",2:"B1"});w.append_uncommitted(2,"dirty")
        self.assertEqual(w.head,2);self.assertEqual(len(w.frames),3)
        self.assertEqual(w.read(2),"B1")
        with self.assertRaises(ValueError):
            w.commit({1:"A2"})
        w.recover();self.assertEqual(len(w.frames),2);self.assertEqual(w.read(2),"B1")

    def test_reset_requires_explicit_completed_backfill_and_reader_release(self):
        w=m.Wal({1:"A0"});w.commit({1:"A1"})
        with self.assertRaises(ValueError):
            w.reset()
        w.checkpoint();w.begin("latest")
        with self.assertRaises(ValueError):
            w.reset()
        w.release("latest");w.reset()
        self.assertEqual(w.read(1),"A1");self.assertEqual(w.snapshot()["model_frames"],0)

    def test_native_sqlite_benign_snapshot_and_checkpoint_oracle(self):
        r=m.sqlite_fixture()
        self.assertEqual(r["old_after_commits"],["A0","B0"])
        self.assertEqual(r["middle_after_commits"],["A1","B0"])
        self.assertEqual(r["latest"],["A2","B2"]);self.assertEqual(r["uncommitted_hidden"],r["latest"])
        self.assertEqual(r["deferred_begin_first_read"],r["latest"])
        self.assertEqual(r["passive_held_reader"][0],0)
        self.assertLess(r["passive_held_reader"][2],r["passive_held_reader"][1])
        self.assertEqual(r["truncate_held_reader"][0],1)
        self.assertEqual(r["truncate_after_release"],[0,0,0])
        self.assertEqual(r["wal_bytes_after_successful_truncate"],0)

    def test_invalid_inputs_and_selected_byte_budget(self):
        w=m.Wal({1:"A0"})
        for data in ({True:"x"},{2:"x"},{1:4},{}):
            with self.assertRaises(ValueError):
                w.commit(data)
        w.begin("old")
        with self.assertRaises(ValueError):
            w.begin("old")
        with self.assertRaises(ValueError):
            w.release("unknown")
        self.assertEqual(m.budget({"frames":3,"page_bytes":4096})["selected_wal_bytes"],12392)
        with self.assertRaises(ValueError):
            m.budget({"frames":True,"page_bytes":4096})

    def test_node_python_trace_parity_and_bad_reset(self):
        result=subprocess.run(["node","-e",JS+"\nconsole.log(JSON.stringify(play("+json.dumps(m.COMMANDS)+")));"],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),m.play(m.COMMANDS))
        bad=[["reader","old"],["commit",{"1":"A1"}],["reset"]]
        result=subprocess.run(["node","-e",JS+"\nplay("+json.dumps(bad)+");"],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn("reset",result.stderr)

    def test_sealed_report_and_cpu_paths(self):
        for n in (64,256,1024,4096):
            a,b=m.timing_paths(n);self.assertEqual(a(),b())
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        for row in latency["rows"]:
            for path in ("indexed","scan"):
                samples=row[path+"_samples_ms"];self.assertEqual(len(samples),7)
                self.assertEqual(sorted(samples)[3],row[path+"_median_ms"])
        native=json.loads((m.HERE/"native_oracle.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency,native))
