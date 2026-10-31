# -*- coding: utf-8 -*-
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPhase144Experiments(unittest.TestCase):
    def _run_and_load(self, exp_dir: str) -> dict:
        script = ROOT / "experiments" / exp_dir / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        summary = ROOT / "experiments" / exp_dir / "results" / "summary.json"
        return json.loads(summary.read_text(encoding="utf-8"))

    def test_gpu_kernels_phase144(self):
        d = self._run_and_load("flashinfer_nvfp4_mla_moe_allreduce_kcp_splitk_phase144")
        self.assertEqual(d["status"], "ok")
        self.assertAlmostEqual(d["nvfp4_mla_reverse_tail_saving_pct"], 24.19, places=2)
        self.assertAlmostEqual(d["moe_finalize_ar_prefetch_speedup_x"], 1.28, places=2)
        self.assertEqual(d["kda_kcp_recompute_comm_saving_pct"], 100.0)

    def test_serving_phase144(self):
        d = self._run_and_load("sglang_detok_vllm_megamoe_trtllm_adp_visualgen_phase144")
        self.assertEqual(d["status"], "ok")
        self.assertAlmostEqual(d["sglang_detok_ipc_saving_pct"], 78.12, places=2)
        self.assertAlmostEqual(d["diffusion_gemma_logits_mem_saving_pct"], 87.5, places=2)
        self.assertAlmostEqual(d["trtllm_adp_kv_skew_reduction_pct"], 68.42, places=2)

    def test_training_kv_phase144(self):
        d = self._run_and_load("megatron_ckpt_dequant_mooncake_egm_nixl_s3rdma_phase144")
        self.assertEqual(d["status"], "ok")
        self.assertAlmostEqual(d["megatron_stream_dequant_mem_saving_pct"], 66.67, places=2)
        self.assertEqual(d["mfsdpv2_partial_grad_comm_saving_pct"], 50.0)
        self.assertAlmostEqual(d["nixl_s3_rdma_speedup_x"], 2.85, places=2)


if __name__ == "__main__":
    unittest.main()
