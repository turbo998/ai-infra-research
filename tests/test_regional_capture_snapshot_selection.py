import json
import statistics
import subprocess
import unittest

from experiments.regional_capture_snapshot_selection import run as m
from experiments.regional_capture_snapshot_selection.render import JS


class RegionalCaptureTests(unittest.TestCase):
    def test_live_and_cloned_values_independent_index_oracle(self):
        for stride in (0,1,2,4):
            source = list(range(32))
            view = m.View(source,3,5,stride)
            expected = source[3:3+5*stride:stride] if stride else [source[3]]*5
            live = m.snapshot([view])[0]
            frozen = m.snapshot([view],True)[0]
            self.assertEqual(frozen.values(),expected)
            source[3] = 900
            self.assertEqual(frozen.values(),expected)
            self.assertEqual(live.values(),[source[3+i*stride] for i in range(5)])

    def test_nested_repeated_identity_preserved(self):
        view = m.View([1,2,3],0,3,1)
        out = m.snapshot({"args":[view],"kwargs":(view,)},True)
        self.assertIs(out["args"][0],out["kwargs"][0])
        self.assertIsNot(out["args"][0].storage,view.storage)

    def test_distinct_shared_storage_rejected_only_for_clone(self):
        source = list(range(10))
        a,b = m.View(source,0,3,1),m.View(source,2,3,1)
        with self.assertRaises(ValueError):
            m.snapshot([a,b],True)
        live = m.snapshot([a,b])
        source[2] = 100
        self.assertEqual(live[0].values()[-1],live[1].values()[0])

    def test_state_alias_guard(self):
        source = list(range(5))
        view = m.View(source,1,3,1)
        with self.assertRaises(ValueError):
            m.snapshot(view,True,{id(source)})
        self.assertIs(m.snapshot(view,False,{id(source)}).storage,source)

    def test_module_aliases_precapture_hook_and_pending_status(self):
        view = m.View([1,2],0,2,1)
        result = m.capture_events([("a",view),("b",view)],{"a":"shared","b":"shared","c":"unused"},True)
        self.assertEqual(result["a"]["calls"],[[1,2],[2,3]])
        self.assertEqual(result["a"],result["b"])
        self.assertIsNone(result["a"]["status"])
        self.assertEqual(result["c"],{"status":"not_reached","calls":[]})
        self.assertEqual(view.values(),[3,4])

    def test_invalid_inputs_and_layout_prefix(self):
        self.assertEqual(m.layout_budget(m.DEFAULT),{"logical_bytes":16,"layout_preserving_buffer_bytes":40})
        for p in ({**m.DEFAULT,"stride":-1},{**m.DEFAULT,"length":0},{**m.DEFAULT,"offset":2**53}):
            with self.assertRaises(ValueError):
                m.layout_budget(p)
        cyclic=[];cyclic.append(cyclic)
        for tree in (cyclic,object()):
            with self.assertRaises(ValueError):
                m.snapshot(tree)

    def test_node_budget_and_mutation_parity(self):
        for stride in (0,1,3):
            p={**m.DEFAULT,"stride":stride}
            result=json.loads(subprocess.check_output(["node","-e",JS+"\nconsole.log(JSON.stringify(simulate("+json.dumps(p)+",50)))"],text=True))
            self.assertEqual(result["budget"],m.layout_budget(p))
            source=list(range(p["offset"]+1+(p["length"]-1)*stride))
            view=m.View(source,p["offset"],p["length"],stride)
            frozen=m.snapshot(view,True);source[p["offset"]]=50
            self.assertEqual(result["detached_after"],view.values())
            self.assertEqual(result["cloned_after"],frozen.values())

    def test_report_and_sealed_medians(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("detach","clone"):
                self.assertEqual(len(row[prefix+"_samples_ms"]),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(row[prefix+"_samples_ms"]))
        self.assertFalse(report["native_compilation_reproduced"])


if __name__ == "__main__":
    unittest.main()
