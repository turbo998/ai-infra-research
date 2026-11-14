from fractions import Fraction
import json
import statistics
import subprocess
import unittest

from experiments.resize_coordinate_contract_selection import run as m
from experiments.resize_coordinate_contract_selection.render import JS


class ResizeCoordinateTests(unittest.TestCase):
    def test_integer_extent_corner_regression(self):
        fixed=m.coordinates(4,scale=Fraction(3,5),mode="align_corners")
        wrong=m.coordinates(4,scale=Fraction(3,5),mode="align_corners",wrong_unrounded_corners=True)
        self.assertEqual(fixed,[0,3])
        self.assertEqual(m.interpolate([1,2,3,4],fixed),[1,4])
        self.assertEqual(m.interpolate([1,2,3,4],wrong),[1,Fraction(22,7)])

    def test_hat_basis_oracle_all_modes(self):
        for n in (1,2,3,7):
            x=[Fraction(i*i-2,3) for i in range(n)]
            for size in (1,2,5,8):
                for mode in m.MODES:
                    coords=m.coordinates(n,size=size,mode=mode)
                    self.assertEqual(m.interpolate(x,coords),m.hat_oracle(x,coords))

    def test_singleton_mode_difference(self):
        x=[1,2,3,4]
        self.assertEqual(m.interpolate(x,m.coordinates(4,size=1)),[Fraction(5,2)])
        self.assertEqual(m.interpolate(x,m.coordinates(4,size=1,mode="pytorch_half_pixel")),[1])
        self.assertEqual(m.coordinates(4,scale=Fraction(2,5),mode="align_corners"),[0])

    def test_sizes_and_explicit_scales_are_not_interchangeable(self):
        size_coords=m.coordinates(4,size=2)
        scale_coords=m.coordinates(4,scale=Fraction(3,5))
        self.assertEqual(len(size_coords),len(scale_coords))
        self.assertNotEqual(size_coords,scale_coords)
        self.assertEqual(m.coordinates(4,size=2,mode="align_corners"),m.coordinates(4,scale=Fraction(3,5),mode="align_corners"))

    def test_nearest_tie_and_edge_replication(self):
        coords=m.coordinates(2,size=3)
        self.assertEqual(m.interpolate([0,10],coords,"round_prefer_floor"),[0,0,10])
        self.assertEqual(m.interpolate([0,10],coords,"round_prefer_ceil"),[0,10,10])
        self.assertEqual(m.interpolate([1,4],[Fraction(-100),Fraction(100)]),[1,4])

    def test_invalid_input_and_budget(self):
        for arguments in ({"n":0,"size":1},{"n":2},{"n":2,"size":3,"scale":Fraction(3,2)},{"n":2,"scale":Fraction(1,10)}):
            with self.assertRaises(ValueError):
                m.extent(**arguments)
        with self.assertRaises(ValueError):
            m.coordinates(2,size=3,mode="unknown")
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"output_side":2**53})

    def test_node_grid_and_payload_parity(self):
        for mode in m.MODES:
            for size in (1,5):
                source=JS+"\nconsole.log(JSON.stringify({result:simulate([0,10,20],"+str(size)+","+json.dumps(mode)+"),budget:budget("+json.dumps(m.DEFAULT)+")}))"
                out=json.loads(subprocess.check_output(["node","-e",source],text=True))
                expected=m.interpolate([0,10,20],m.coordinates(3,size=size,mode=mode))
                for a,b in zip(out["result"]["values"],expected):
                    self.assertAlmostEqual(a,float(b),places=12)
                self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_cached_float_paths(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("cached","recompute"):
                self.assertEqual(len(row[prefix+"_samples_ms"]),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(row[prefix+"_samples_ms"]))
        a,b=m.float_paths(8)
        self.assertEqual(a(),b())
        self.assertIsNone(report["vision_model_accuracy"])


if __name__=="__main__":
    unittest.main()
