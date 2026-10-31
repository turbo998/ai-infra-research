# -*- coding: utf-8 -*-
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPhase145Experiments(unittest.TestCase):
    def _run_and_load(self, exp_dir: str) -> dict:
        script = ROOT / "experiments" / exp_dir / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        summary = ROOT / "experiments" / exp_dir / "results" / "summary.json"
        return json.loads(summary.read_text(encoding="utf-8"))

    def test_gpu_kernels_phase145(self):
        d = self._run_and_load("flashinfer_minimaxh3_sm120_sdsa_ulysses_tilelang_phase145")
        self.assertEqual(d["status"], "ok")
        self.assertAlmostEqual(d["sm120_minimax_h3_varlen_speedup_x"], 1.34, places=2)
        self.assertAlmostEqual(d["cutlass_no_reinit_launch_saving_pct"], 61.54, places=2)
        self.assertAlmostEqual(d["sdsa_key_union_kv_saving_pct"], 62.5, places=2)

    def test_serving_phase145(self):
        d = self._run_and_load("sglang_qwen38_dsv4_vllm_swa_trtllm_helix_dspark_phase145")
        self.assertEqual(d["status"], "ok")
        self.assertAlmostEqual(d["sglang_qwen38_hc_gdn_speedup_x"], 1.45, places=2)
        self.assertAlmostEqual(d["vllm_dsv41_swa_meta_saving_pct"], 68.75, places=2)
        self.assertAlmostEqual(d["trtllm_dspark_ragged_graph_speedup_x"], 1.31, places=2)

    def test_training_kv_phase145(self):
        d = self._run_and_load("megatron_mfsdp_mtp_mooncake_datadirect_nixl_libfabric_phase145")
        self.assertEqual(d["status"], "ok")
        self.assertAlmostEqual(d["megatron_decoupled_mfsdp_comm_saving_pct"], 42.86, places=2)
        self.assertAlmostEqual(d["mooncake_data_direct_pci_gain_pct"], 38.46, places=2)
        self.assertAlmostEqual(d["nixl_libfabric_peer_scale_factor_x"], 16.0, places=1)


if __name__ == "__main__":
    unittest.main()
