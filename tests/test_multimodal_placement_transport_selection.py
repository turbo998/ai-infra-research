import json
import random
import subprocess
import unittest

from experiments.multimodal_placement_transport_selection import run as m
from experiments.multimodal_placement_transport_selection.render import JS,cards


class PlacementTransportTests(unittest.TestCase):
    def test_gpu_preprocessing_with_cpu_transport_caps_auto(self):
        r=m.evaluate(m.DEFAULT)
        self.assertEqual(r["candidate_fast_device"],"cuda:3")
        self.assertTrue(r["preprocessing_contends"])
        self.assertEqual((r["auto_processor_workers"],r["naive_transport_only_auto_workers"]),(1,8))
        self.assertFalse(r["keep_features_on_device_policy"])

    def test_cpu_rl_disabled_and_non_hf_placement(self):
        for changes in ({"platform":"cpu"},{"rl_cpu":True},{"fast_disabled":True},{"hf_image_processor":False}):
            r=m.evaluate({**m.DEFAULT,**changes,"transport":"cuda_ipc"})
            self.assertFalse(r["preprocessing_contends"])
            self.assertEqual(r["auto_processor_workers"],8)
            self.assertTrue(r["keep_features_on_device_policy"])
        r=m.evaluate({**m.DEFAULT,"platform":"cpu","declared_workers":None})
        self.assertEqual(r["auto_processor_workers"],2)

    def test_npu_unset_xpu_and_instance_identity(self):
        r=m.evaluate({**m.DEFAULT,"platform":"npu","processor_class":"Glm4vProcessor"})
        self.assertIsNone(r["candidate_fast_device"]);self.assertTrue(r["preprocessing_contends"])
        self.assertEqual(r["auto_processor_workers"],1)
        self.assertEqual(m.evaluate({**m.DEFAULT,"platform":"xpu"})["candidate_fast_device"],"xpu")
        self.assertEqual(m.evaluate({**m.DEFAULT,"base_gpu_id":7})["candidate_fast_device"],"cuda:7")

    def test_manual_override_and_skip_pool_are_not_auto(self):
        r=m.evaluate({**m.DEFAULT,"explicit_workers":4})
        self.assertEqual((r["auto_processor_workers"],r["selected_processor_workers"]),(1,4))
        r=m.evaluate({**m.DEFAULT,"explicit_workers":4,"skip_pool":True,"transport":"cuda_ipc"})
        self.assertEqual(r["selected_processor_workers"],1);self.assertFalse(r["ipc_pool_configured"])

    def test_budget_divides_tokenizer_processes_not_threads(self):
        r=m.evaluate({**m.DEFAULT,"transport":"cuda_ipc","explicit_workers":17})
        self.assertEqual((r["per_tokenizer_budget_bytes"],r["aggregate_budget_split_bytes"],r["unused_budget_bytes"]),(341,1023,2))
        self.assertEqual(r["naive_full_budget_per_tokenizer_bytes"],3075)
        small=m.evaluate({**m.DEFAULT,"transport":"cuda_ipc","budget_bytes":1})
        self.assertEqual((small["per_tokenizer_budget_bytes"],small["unused_budget_bytes"]),(0,1))
        self.assertIsNone(small["native_zero_share_pool_acceptance"])

    def test_vmm_start_policy_and_invalid_configuration(self):
        r=m.evaluate({**m.DEFAULT,"transport":"cuda_vmm"})
        self.assertTrue(r["keep_features_on_device_policy"]);self.assertFalse(r["cuda_ipc_policy"])
        self.assertEqual(r["cpu_executor_start_method_policy"],"spawn")
        for change in ({"transport":"typo"},{"tokenizer_workers":0},{"budget_bytes":0},{"explicit_workers":0},{"base_gpu_id":True}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.evaluate({**m.DEFAULT,**change})

    def test_random_projection_node_parity_and_hoisting(self):
        rng=random.Random(156);inputs=[m.DEFAULT]
        for _ in range(48):
            inputs.append({**m.DEFAULT,"platform":rng.choice(["cpu","npu","xpu","cuda_alike","other"]),
                           "processor_class":rng.choice(["GenericProcessor","Glm4vProcessor"]),
                           "transport":rng.choice(m.TRANSPORTS),"rl_cpu":bool(rng.randrange(2)),
                           "fast_disabled":bool(rng.randrange(2)),"tokenizer_workers":rng.randrange(1,9),
                           "budget_bytes":rng.randrange(1,2049),"explicit_workers":rng.choice([None,1,4])})
        script=JS+"\nconsole.log(JSON.stringify("+json.dumps(inputs)+".map(evaluate)));"
        result=subprocess.run(["node"],input=script,capture_output=True,text=True,check=True)
        expected=[m.evaluate(p) for p in inputs];self.assertEqual(json.loads(result.stdout),expected)
        for p,r in zip(inputs,expected):
            self.assertLessEqual(r["aggregate_budget_split_bytes"],p["budget_bytes"])
            self.assertLess(r["unused_budget_bytes"],p["tokenizer_workers"])
        for n in (3,64,256):
            funcs=m.timing_paths(n);self.assertEqual(funcs[0](),funcs[1]())

    def test_sealed_matrix_and_csp(self):
        latency=json.loads((m.HERE/"latency.json").read_text());r=json.loads((m.HERE/"report.json").read_text())
        self.assertEqual(r,m.build_report(latency))
        self.assertEqual(r["summary"],{"cases":18,"naive_transport_only_auto_misses":10})
        self.assertIsNone(r["native_sglang_verified"])
        text=cards(r);self.assertEqual(text.count('class="card"'),7);self.assertNotIn("<script",text)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",text)
