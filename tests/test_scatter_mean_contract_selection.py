import json
import statistics
import subprocess
import unittest

from experiments.scatter_mean_contract_selection import run as m
from experiments.scatter_mean_contract_selection.render import JS


class ScatterMeanTests(unittest.TestCase):
    def test_exact_coordinate_oracle_for_every_reduction_and_depth(self):
        for shape in ([2],[2,3],[2,2,2]):
            data=list(range(m.math.prod(shape)))
            for k in range(len(shape)+1):
                row=[0]*k;indices=[row,list(row)]
                width=m.math.prod(shape[k:])
                updates=list(range(-width,width))
                for reduction in ("add","mul","max","min","mean"):
                    self.assertEqual(m.scatter(shape,data,indices,updates,k,reduction),m.independent_destination_oracle(shape,data,indices,updates,k,reduction))

    def test_negative_alias_means_include_original_once(self):
        result=m.scatter([2,2],[10,-4,30,40],[[0],[-2]],[2,-3,8,-2],1)
        self.assertEqual(result,[6,-3,30,40])
        self.assertNotEqual(result,[5,-3,30,40])

    def test_reversal_and_joint_alias_normalization(self):
        args=([2,2],[10,-4,30,40])
        self.assertEqual(m.scatter(*args,[[0],[-2]],[2,-3,8,-2],1),m.scatter(*args,[[-2],[0]],[8,-2,2,-3],1))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            m.scatter(*args,[[0],[-2]],[2,-3,8,-2],1,"none")

    def test_whole_tensor_and_single_element_updates(self):
        self.assertEqual(m.scatter([2],[10,-4],[[],[]],[2,-3,8,-2],0),[6,-3])
        self.assertEqual(m.scatter([2,2],[1,2,3,4],[[1,-1]],[10],2,"none"),[1,2,3,10])
        self.assertEqual(m.scatter([2],[1,2],[],[],1),[1,2])

    def test_negative_floor_and_wide_exact_accumulator(self):
        self.assertEqual(m.scatter([1],[-2],[[0]],[-3],1),[-3])
        large=2**63-1
        self.assertEqual(m.scatter([1],[large],[[0],[0]],[large,large],1),[large])
        self.assertEqual(m.scatter([1],[-large],[[0],[0]],[-large,-large],1),[-large])

    def test_invalid_shapes_bounds_opset_and_budget(self):
        for args in (([2],[1,2],[[2]],[3],1),([2],[1,2],[[-3]],[3],1),([2,2],[1,2,3,4],[[0]],[5],1)):
            with self.assertRaises(ValueError):
                m.scatter(*args)
        with self.assertRaisesRegex(ValueError,"opset29"):
            m.scatter([1],[1],[[0]],[3],1,opset=28)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"rows":2**53})

    def test_node_integer_outputs_budget_and_rejections(self):
        for k,indices,updates in ((0,[[],[]],[2,-3,8,-2,8,7,6,5]),(1,[[0],[-2]],[2,-3,8,-2]),(2,[[0,0],[0,-2]],[2,8])):
            code=JS+"\nconsole.log(JSON.stringify({result:simulate([2,2],[10,-4,30,40],"+json.dumps(indices)+","+json.dumps(updates)+","+str(k)+",'mean',29),budget:budget("+json.dumps(m.DEFAULT)+")}))"
            out=json.loads(subprocess.check_output(["node","-e",code],text=True))
            self.assertEqual(out["result"]["output_flat"],m.scatter([2,2],[10,-4,30,40],indices,updates,k))
            self.assertEqual(out["budget"],m.budget(m.DEFAULT))
        code=JS+"\ntry{simulate([2,2],[10,-4,30,40],[[0],[-2]],[2,-3,8,-2],1,'none',29);process.exit(1)}catch(e){console.log(e.message)}"
        self.assertIn("duplicate",subprocess.check_output(["node","-e",code],text=True))

    def test_sealed_report_and_timing_paths(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        self.assertEqual(report["case"]["wrong_repeated_pair_mean"][0],7)
        self.assertEqual(report["case"]["wrong_reversed_pair_mean"][0],5)
        for row in latency["rows"]:
            for prefix in ("grouped","repeated"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(256);self.assertEqual(a(),b())


if __name__=="__main__":
    unittest.main()
