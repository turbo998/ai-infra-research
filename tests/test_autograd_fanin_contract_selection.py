from fractions import Fraction
import itertools
import json
import subprocess
import unittest

from experiments.autograd_fanin_contract_selection import run as m
from experiments.autograd_fanin_contract_selection.render import JS


class AutogradFaninTests(unittest.TestCase):
    def test_rounding_against_exact_encoded_reference(self):
        exact=sum(Fraction(m.f32(v)) for v in m.VALUES)
        self.assertEqual(exact,1)
        self.assertEqual(m.fold(m.VALUES),0)
        self.assertEqual(m.fold(m.VALUES,True),float(exact))
        for values in itertools.permutations(m.VALUES):
            self.assertEqual(m.fold(values,True),float(exact))
        self.assertGreater(len({m.fold(values) for values in itertools.permutations(m.VALUES)}),1)

    def test_same_slot_zero_and_undefined(self):
        task=m.BackwardSlots(True)
        task.receive("x:0",None,"inactive")
        task.receive("x:0",0,"first-defined")
        task.receive("x:0",None,"undefined")
        with self.assertRaisesRegex(ValueError,"second producer=zero"):
            task.receive("x:0",0,"zero")
        self.assertEqual(task.slots,{"x:0":0})

    def test_distinct_slots_explicit_combine_and_pruning(self):
        result,parent=m.explicit_fanin(m.VALUES)
        self.assertEqual(result,1)
        self.assertEqual(parent,{"upstream:0":1})
        task=m.BackwardSlots(True)
        task.receive("h:0",2,"only-active-branch")
        self.assertEqual(task.slots,{"h:0":2})
        task.receive("h:1",3,"another-slot")
        self.assertEqual(task.slots["h:1"],3)

    def test_separate_task_leaf_accumulation(self):
        leaf=0
        for value in (1,2,-1):
            task=m.BackwardSlots(True)
            task.receive("leaf:0",value,"only-use")
            leaf=m.f32(leaf+task.slots["leaf:0"])
        self.assertEqual(leaf,2)
        task=m.BackwardSlots(False)
        task.receive("h:0",1,"a")
        task.receive("h:0",2,"b")
        self.assertEqual(task.slots["h:0"],3)

    def test_parent_inheritance_and_compiled_contract(self):
        parent=m.BackwardSlots(True)
        child=m.BackwardSlots(False,parent)
        self.assertTrue(child.strict)
        child.receive("h:0",1,"a")
        with self.assertRaises(ValueError):
            child.receive("h:0",2,"b")
        with self.assertRaises(NotImplementedError):
            m.BackwardSlots(False,parent,compiled=True)
        self.assertFalse(m.BackwardSlots(False,compiled=True).strict)

    def test_invalid_floats_flags_and_budget(self):
        for value in (True,float("nan"),float("inf"),1e40,"1"):
            with self.assertRaises(ValueError):
                m.f32(value)
        for args in ((1,None,False),(False,1,False)):
            with self.assertRaises(ValueError):
                m.BackwardSlots(*args)
        for p in ({**m.DEFAULT,"elements":0},{**m.DEFAULT,"branches":True},{**m.DEFAULT,"elements":2**53-1}):
            with self.assertRaises(ValueError):
                m.estimate_budget(p)

    def test_node_fold_and_budget_parity(self):
        cases=[m.VALUES,list(reversed(m.VALUES)),[0,1,2],[0.25,0.5,-0.75]]
        source="const DEFAULT="+json.dumps(m.DEFAULT)+";\n"+JS+"\nconsole.log(JSON.stringify({folds:"+json.dumps(cases)+".map(calculate),budget:estimate(DEFAULT)}));"
        result=json.loads(subprocess.check_output(["node","-e",source],text=True))
        for values,actual in zip(cases,result["folds"]):
            self.assertEqual(actual["float32_left_fold"],m.fold(values))
            self.assertEqual(actual["wide_final_float32"],m.fold(values,True))
        self.assertEqual(result["budget"],m.estimate_budget(m.DEFAULT))
        source="const DEFAULT="+json.dumps(m.DEFAULT)+";\n"+JS+"\ntry{calculate([Infinity]);process.exit(1)}catch(e){console.log(e.message)}"
        self.assertTrue(subprocess.check_output(["node","-e",source],text=True).strip())

    def test_sealed_report_and_sample_medians(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("wide","rounded"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],sorted(samples)[3])
        self.assertFalse(report["full_dag_engine_reproduced"])


if __name__=="__main__":
    unittest.main()
