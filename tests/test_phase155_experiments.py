# -*- coding: utf-8 -*-
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase155Experiments(unittest.TestCase):
    def test_exp1_vsa_sdsa_fp4_triton_tilelang(self) -> None:
        exp_dir = ROOT / "experiments" / "flashinfer_vsa_sdsa_fp4_triton_gluon_tilelang_sm120_phase155"
        subprocess.check_call([sys.executable, str(exp_dir / "run.py"), "--check"])
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 155)
        self.assertGreater(data["sdsa_token_sparse_mqa"]["sdsa_decode_speedup"], 6.5)
        self.assertEqual(data["triton_and_tilelang_codegen"]["gluon_unreleased_accum_leaks"], 0)

    def test_exp2_sglang_pd_poll_vllm_dsv41_kv_gather_trtllm(self) -> None:
        exp_dir = ROOT / "experiments" / "sglang_pd_poll_vllm_dsv41_kv_gather_trtllm_mamba_phase155"
        subprocess.check_call([sys.executable, str(exp_dir / "run.py"), "--check"])
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["dsv41_kv_source_group_gather"]["kv_gather_reduction_pct"], 75.0)
        self.assertGreater(data["sglang_pd_poll_and_ipc"]["pd_poll_overlap_speedup"], 3.5)
        self.assertGreater(data["trtllm_mamba_and_fmha"]["mamba_prep_speedup"], 3.0)

    def test_exp3_megatron_gtp_owner_p2p_mooncake_lmcache_apc(self) -> None:
        exp_dir = ROOT / "experiments" / "megatron_gtp_owner_p2p_mooncake_mnnvl_lmcache_apc_phase155"
        subprocess.check_call([sys.executable, str(exp_dir / "run.py"), "--check"])
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["megatron_gtp_and_owner_p2p"]["gtp_activation_savings_pct"], 60.0)
        self.assertEqual(data["lmcache_apc_prefix_skip"]["lmcache_apc_skip_speedup"], 4.0)
        self.assertEqual(data["mooncake_efa_and_mnnvl"]["mnnvl_cumem_import_einval_errors"], 0)


if __name__ == "__main__":
    unittest.main()
