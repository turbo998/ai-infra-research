from fractions import Fraction
import json
import statistics
import subprocess
import unittest

from experiments.nms_optional_contract_selection import run as m
from experiments.nms_optional_contract_selection.render import JS


class NMSOptionalTests(unittest.TestCase):
    def test_exact_iou_against_integer_cell_oracle(self):
        boxes=[[0,0,2,2],[0,1,2,3],[3,3,1,1],[0,0,0,1]]
        def cells(box):
            y1,x1,y2,x2=map(int,m.canonical(box))
            return {(y,x) for y in range(y1,y2) for x in range(x1,x2)}
        for a in boxes:
            for b in boxes:
                sa,sb=cells(a),cells(b);union=len(sa|sb)
                expected=Fraction(len(sa&sb),union) if union else Fraction()
                self.assertEqual(m.iou(m.canonical(a),m.canonical(b)),expected)

    def test_independent_candidate_removal_oracle(self):
        boxes=[[0,i,2,i+2] for i in range(6)]
        scores=[[Fraction(6-i,10) for i in range(6)],[Fraction(i,10) for i in range(6)]]
        for threshold in (Fraction(),Fraction(1,3),Fraction(1)):
            for maximum in (0,1,3,10):
                self.assertEqual(m.nms(boxes,scores,maximum,threshold),m.candidate_removal_oracle(boxes,scores,maximum,threshold))

    def test_absent_score_not_explicit_zero(self):
        boxes=[[0,0,1,1],[0,2,1,3],[0,4,1,5]];scores=[[-1,0,1]]
        self.assertEqual(m.nms(boxes,scores,3,1),[[0,0,2],[0,0,1],[0,0,0]])
        self.assertEqual(m.nms(boxes,scores,3,1,0),[[0,0,2]])
        self.assertEqual(m.nms(boxes,scores,3,1,[]),m.nms(boxes,scores,3,1))

    def test_optional_cap_wrappers_empty_and_per_class(self):
        boxes=[[0,0,1,1],[0,2,1,3]];scores=[[2,1],[1,2]]
        self.assertEqual(m.nms(boxes,scores),[])
        self.assertEqual(m.nms(boxes,scores,[]),[])
        self.assertEqual(m.nms(boxes,scores,-1),[])
        self.assertEqual(m.nms(boxes,scores,[1],[1]),[[0,0,0],[0,1,1]])
        with self.assertRaises(ValueError):
            m.optional_scalar([1,2])

    def test_iou_boundary_center_and_tie_order(self):
        boxes=[[0,0,2,2],[0,1,2,3]];scores=[[1,1]]
        self.assertEqual(m.nms(boxes,scores,2,Fraction(1,3)),[[0,0,0],[0,0,1]])
        self.assertEqual(m.nms(boxes,scores,2,Fraction(1,3),wrong_equal=True),[[0,0,0]])
        center=[[1,1,2,2],[2,1,2,2]]
        self.assertEqual(m.nms(center,scores,2,Fraction(1,3),center=True),m.nms(boxes,scores,2,Fraction(1,3)))

    def test_invalid_shapes_geometry_and_budget(self):
        with self.assertRaises(ValueError):
            m.nms([[0,0,1,1]],[[1,2]],1)
        with self.assertRaises(ValueError):
            m.canonical([0,0,-1,2],True)
        with self.assertRaises(ValueError):
            m.nms([[0,0,1,1]],[[1]],1,2)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"classes":2**53})

    def test_node_selection_and_budget_parity(self):
        boxes=[[0,0,2,2],[0,1,2,3],[10,10,12,12]];scores=[[9,8,-1],[1,2,0]]
        for threshold in (None,0):
            source=JS+"\nconsole.log(JSON.stringify({result:simulate("+json.dumps(boxes)+","+json.dumps(scores)+",2,1/3,"+json.dumps(threshold)+"),budget:budget("+json.dumps(m.DEFAULT)+")}))"
            out=json.loads(subprocess.check_output(["node","-e",source],text=True))
            self.assertEqual(out["result"]["selected_indices"],m.nms(boxes,scores,2,Fraction(1,3),threshold))
            self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_float_paths(self):
        timing=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(timing))
        for row in timing["rows"]:
            for prefix in ("cached","normalize"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.float_paths(8)
        self.assertEqual(a(),b())
        self.assertIsNone(report["detection_quality"])


if __name__=="__main__":
    unittest.main()
