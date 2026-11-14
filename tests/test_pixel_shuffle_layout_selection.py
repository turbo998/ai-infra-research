import json
import statistics
import subprocess
import unittest

from experiments.pixel_shuffle_layout_selection import run as m
from experiments.pixel_shuffle_layout_selection.render import JS


class PixelShuffleLayoutTests(unittest.TestCase):
    def test_exact_crd_vs_dcr_fixture(self):
        x=list(range(16));crd=m.depth_to_space(x,m.CASE);dcr=m.depth_to_space(x,{**m.CASE,"mode":"DCR"})
        self.assertEqual(crd,[0,2,1,3,4,6,5,7,8,10,9,11,12,14,13,15])
        self.assertEqual(dcr,[0,4,1,5,8,12,9,13,2,6,3,7,10,14,11,15])
        self.assertEqual(sum(a==b for a,b in zip(crd,dcr)),4)

    def test_independent_scatter_oracle_multiple_shapes(self):
        for b in (1,2,3):
            for co in (1,2,3):
                for mode in ("DCR","CRD"):
                    p={**m.CASE,"n":2,"c":co*b*b,"h":2,"w":3,"blocksize":b,"mode":mode}
                    values=list(range(m.budget(p)["elements_preserved"]))
                    self.assertEqual(m.depth_to_space(values,p),m.scatter_oracle(values,p))
                    self.assertEqual(sorted(m.permutation(p)),list(range(len(values))))

    def test_matching_inverse_and_adjoint_identity(self):
        for mode in ("DCR","CRD"):
            p={**m.CASE,"mode":mode};x=list(range(16));g=list(range(1,17))
            y=m.depth_to_space(x,p)
            self.assertEqual(m.space_to_depth(y,p),x)
            self.assertEqual(sum(a*b for a,b in zip(y,g)),sum(a*b for a,b in zip(x,m.space_to_depth(g,p))))

    def test_single_output_channel_hides_mode_difference(self):
        p={**m.CASE,"c":4};values=list(range(8))
        self.assertEqual(m.depth_to_space(values,p),m.depth_to_space(values,{**p,"mode":"DCR"}))
        self.assertNotEqual(m.space_to_depth(m.depth_to_space(list(range(16)),m.CASE),{**m.CASE,"mode":"DCR"}),list(range(16)))

    def test_direct_single_op_downgrade_gate(self):
        self.assertTrue(m.direct_space_to_depth_export_mode("DCR",27)["direct_single_op_allowed"])
        self.assertTrue(m.direct_space_to_depth_export_mode("CRD",28)["direct_single_op_allowed"])
        with self.assertRaises(ValueError):
            m.direct_space_to_depth_export_mode("CRD",27)

    def test_divisibility_exact_bounds_and_logical_payload(self):
        out=m.budget(m.DEFAULT)
        self.assertEqual(out["output_shape"],[1,64,256,256])
        self.assertEqual(out["input_payload_bytes"],16777216)
        self.assertEqual(out["selected_one_read_one_write_bytes"],33554432)
        for patch in ({"c":7},{"blocksize":0},{"mode":"bad"},{"n":True},{"scalar_bytes":3}):
            with self.assertRaises(ValueError):
                m.budget({**m.CASE,**patch})

    def test_node_python_coordinate_and_budget_parity(self):
        for mode in ("CRD","DCR"):
            p={**m.CASE,"mode":mode}
            actual=json.loads(subprocess.check_output(["node","-e",JS+"\nconsole.log(JSON.stringify(simulate("+json.dumps(p)+")))"],text=True))
            self.assertEqual(actual["output_values"],m.depth_to_space(list(range(16)),p))
            self.assertEqual(actual["inverse_values"],list(range(16)))
            self.assertEqual(actual["contract"],m.budget(p))

    def test_sealed_report_and_timing_records(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("preplanned","rederive"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(256);self.assertEqual(a(),b())
        self.assertIsNone(report["native_gpu_latency_ms"])


if __name__=="__main__":
    unittest.main()
