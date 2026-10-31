# -*- coding: utf-8 -*-
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPhase146Experiments(unittest.TestCase):
    def _run_and_load(self, exp_dir: str) -> dict:
        script = ROOT / "experiments" / exp_dir / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        summary = ROOT / "experiments" / exp_dir / "results" / "summary.json"
        return json.loads(summary.read_text(encoding="utf-8"))

    def test_gpu_kernels_phase146(self):
        d = self._run_and_load("flashinfer_moe_fencewarp_rmsnorm_sm103_mhc_tilelang_phase146")
        self.assertEqual(d["status"], "ok")
        self.assertAlmostEqual(d["moe_allreduce_one_token_saving_pct"], 25.0, places=2)
        self.assertAlmostEqual(d["rmsnorm_sm107_wgrad_speedup_x"], 1.28, places=2)
        self.assertAlmostEqual(d["tilelang_autotune_input_reuse_saving_pct"], 56.25, places=2)

    def test_serving_phase146(self):
        d = self._run_and_load("sglang_quark_dspark_vllm_flashkda_swa_trtllm_mx_phase146")
        self.assertEqual(d["status"], "ok")
        self.assertAlmostEqual(d["sglang_npu_w8a8s_dspark_speedup_x"], 1.48, places=2)
        self.assertAlmostEqual(d["vllm_flashkda_strided_saving_pct"], 28.12, places=2)
        self.assertAlmostEqual(d["vllm_dsv41_swa_replay_ttft_saving_pct"], 46.88, places=2)

    def test_training_kv_phase146(self):
        d = self._run_and_load("megatron_mfsdp_weakkey_mooncake_rdma_pipe_dynamo_nixl_phase146")
        self.assertEqual(d["status"], "ok")
        self.assertAlmostEqual(d["dynamo_sourceless_vt_saving_pct"], 31.25, places=2)
        self.assertAlmostEqual(d["mooncake_rdma_slice_pipeline_saving_pct"], 36.0, places=2)
        self.assertAlmostEqual(d["nixl_locality_vmm_latency_saving_pct"], 26.67, places=2)


if __name__ == "__main__":
    unittest.main()
