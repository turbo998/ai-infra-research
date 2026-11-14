import json
import statistics
import subprocess
import unittest

from experiments.transpose_output_padding_selection import run as m
from experiments.transpose_output_padding_selection.render import JS


class TransposeOutputPaddingTests(unittest.TestCase):
    def test_padding_selects_nonzero_boundary_not_zero_append(self):
        x,k=[1,2],[1,2,3]
        self.assertEqual(m.transpose(x,k,{**m.PARAMS,"output_padding":0}),[2,5,4])
        self.assertEqual(m.transpose(x,k,m.PARAMS),[2,5,4,6])

    def test_independent_dense_matrix_oracle_across_domains(self):
        for n in (1,2,4):
            for stride in (1,2,3):
                for op in range(stride):
                    for dilation in (1,2):
                        p={"stride":stride,"padding":0,"dilation":dilation,"output_padding":op}
                        x=list(range(1,n+1));kernel=[2,-1,3]
                        self.assertEqual(m.transpose(x,kernel,p),m.matrix_transpose_apply(m.matrix_oracle(n,kernel,p),x))

    def test_adjoint_not_inverse(self):
        x,k=[1,2],[1,2,3];z=[1,2,3,4];a=m.matrix_oracle(2,k,m.PARAMS)
        az=[sum(v*w for v,w in zip(row,z)) for row in a]
        self.assertEqual(az,[8,20])
        self.assertEqual(sum(v*w for v,w in zip(az,x)),sum(v*w for v,w in zip(z,m.transpose(x,k,m.PARAMS))))
        self.assertEqual(m.transpose(az,k,m.PARAMS),[16,44,40,60])
        self.assertNotEqual(m.transpose(az,k,m.PARAMS),z)

    def test_requested_output_size_interval_and_subset_policy(self):
        self.assertEqual(m.requested_output_padding(2,3,m.PARAMS,4),1)
        self.assertEqual(m.requested_output_padding(1,2,m.PARAMS,1),1)
        with self.assertRaises(ValueError):
            m.requested_output_padding(2,3,m.PARAMS,5)
        with self.assertRaises(ValueError):
            m.output_length(2,3,{**m.PARAMS,"output_padding":2})

    def test_zero_and_linearity_controls(self):
        p=m.PARAMS;k=[1,-2,3];a=[1,2];b=[-3,4]
        self.assertEqual(m.transpose([0,0],k,p),[0]*4)
        self.assertEqual(m.transpose(a,[0,0,0],p),[0]*4)
        self.assertEqual(m.transpose([x+y for x,y in zip(a,b)],k,p),[x+y for x,y in zip(m.transpose(a,k,p),m.transpose(b,k,p))])

    def test_grouped_budget_and_explicit_bad_inputs(self):
        self.assertEqual(m.budget(m.DEFAULT),{"output_length":256,"input_payload_bytes":262144,"output_payload_bytes":262144,"weight_payload_bytes":6144,"selected_read_input_and_weights_once_write_output_bytes":530432,"note":"grouped logical tensor payload only;scalar evaluator is onechannel;no native traffic,workspace,allocator or convolution algorithm"})
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"groups":3})
        with self.assertRaises(ValueError):
            m.transpose([True,1],[1,2,3],m.PARAMS)
        with self.assertRaises(ValueError):
            m.output_length(1,1,{**m.PARAMS,"padding":10})

    def test_node_python_scalar_and_budget_parity(self):
        result=json.loads(subprocess.check_output(["node","-e",JS+"\nconsole.log(JSON.stringify({result:simulate([1,2],[1,2,3],"+json.dumps(m.PARAMS)+"),budget:budget("+json.dumps(m.DEFAULT)+")}))"],text=True))
        self.assertEqual(result["result"]["output_values"],m.transpose([1,2],[1,2,3],m.PARAMS))
        self.assertEqual(result["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_timing_records(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("scatter","dense"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(32);self.assertEqual(a(),b())
        self.assertIsNone(report["native_pytorch_parity"])


if __name__=="__main__":
    unittest.main()
