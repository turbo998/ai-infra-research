import json
import random
import subprocess
import unittest

from experiments.rocm_fd_cache_capability_selection import run as m
from experiments.rocm_fd_cache_capability_selection.render import JS,cards


class CacheCapabilityTests(unittest.TestCase):
    def test_rocm_fd_keeps_config_without_registration(self):
        old=m.capture(m.DEFAULT,True);new=m.resolve(m.DEFAULT)
        self.assertEqual(old["status"],"rejected")
        self.assertTrue(new["configuration_enabled"])
        self.assertTrue(new["needs_single_physical_allocation_per_segment"])
        self.assertIsNone(new["native_startup_verified"])

    def test_registration_and_combined_fd_remain_blocked(self):
        r=m.resolve({**m.DEFAULT,"capabilities":["REGISTERS"]})
        self.assertFalse(r["configuration_enabled"])
        with self.assertRaises(m.PolicyError):
            m.resolve({**m.DEFAULT,"capabilities":["REGISTERS"],"requested":True})
        r=m.capture({**m.DEFAULT,"capabilities":["REGISTERS","EXPORTS_FD"]})
        self.assertEqual(r["reason_code"],"FD_REQUIRES_EXTENSIBLE")

    def test_fd_explicit_false_and_other_override_priority(self):
        for change,detail in (({"requested":False},"explicitly disabled"),({"memory_override":True},"MEMORY_OVERRIDE"),
                              ({"blocks_override":True},"BLOCKS_OVERRIDE"),({"elastic_ep":True},"ELASTIC_EP"),
                              ({"custom_pool":True},"CUSTOM_POOL"),({"capabilities":["EXPORTS_FD","LEGACY_IPC"]},"LEGACY_IPC")):
            r=m.capture({**m.DEFAULT,**change})
            self.assertEqual(r["status"],"rejected");self.assertEqual(r["detail"],detail)

    def test_fabric_auto_explicit_and_fd_precedence(self):
        p={**m.DEFAULT,"capabilities":["EXPORTS_FABRIC"]}
        self.assertFalse(m.resolve(p)["configuration_enabled"])
        self.assertTrue(m.resolve({**p,"requested":True})["configuration_enabled"])
        self.assertEqual(m.resolve({**p,"requested":True})["warnings"],["NO_IMEX_LOCAL_TRANSFER_TCP_WARNING"])
        self.assertTrue(m.resolve({**p,"imex":True})["configuration_enabled"])
        self.assertTrue(m.resolve({**p,"capabilities":["EXPORTS_FD","EXPORTS_FABRIC"]})["configuration_enabled"])

    def test_runtime_probe_failure_differs_for_fd_and_ordinary(self):
        fd=m.capture({**m.DEFAULT,"driver_probe":"unsupported"})
        self.assertEqual((fd["reason_code"],fd["stage"],fd["configuration_enabled"]),("FD_DRIVER_REQUIRED","runtime",True))
        plain=m.resolve({**m.DEFAULT,"capabilities":[],"driver_probe":"unsupported","requested":True})
        self.assertTrue(plain["configuration_enabled"]);self.assertFalse(plain["runtime_enabled_in_model"])
        self.assertEqual(plain["runtime_action"],"model_runtime_fallback_ordinary_allocator")

    def test_invalid_schema_and_non_cuda_configuration(self):
        for change in ({"requested":1},{"capabilities":["typo"]},{"capabilities":["EXPORTS_FD","EXPORTS_FD"]},{"driver_probe":True}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.capture({**m.DEFAULT,**change})
        r=m.capture({**m.DEFAULT,"cuda_alike":False})
        self.assertEqual(r["detail"],"NON_CUDA_ALIKE")

    def test_random_capabilities_node_parity_and_classification(self):
        rng=random.Random(157);inputs=[m.DEFAULT]
        for _ in range(64):
            inputs.append({**m.DEFAULT,"capabilities":[v for v in m.BITS if rng.randrange(2)],
                           "rocm":bool(rng.randrange(2)),"requested":rng.choice([None,True,False]),
                           "imex":bool(rng.randrange(2)),"driver_probe":rng.choice(["unknown","supported","unsupported"])})
        script=JS+"\nconsole.log(JSON.stringify("+json.dumps(inputs)+".map(p=>[capture(p,true),capture(p)])));"
        result=subprocess.run(["node"],input=script,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[[m.capture(p,True),m.capture(p)] for p in inputs])
        for n in (16,64,256):
            funcs=m.classification_paths(n);self.assertEqual(funcs[0](),funcs[1]())

    def test_sealed_semantic_matrix_and_csp(self):
        latency=json.loads((m.HERE/"latency.json").read_text());r=json.loads((m.HERE/"report.json").read_text())
        self.assertEqual(r,m.build_report(latency))
        self.assertEqual(r["summary"],{"cases":12,"changed_configuration_results":5,"before_enabled":2,"after_enabled":7})
        self.assertIsNone(r["native_rocm_fd_export_verified"])
        text=cards(r);self.assertEqual(text.count('class="card"'),7);self.assertNotIn("<script",text)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",text)
