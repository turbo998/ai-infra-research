# -*- coding: utf-8 -*-
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase154Experiments(unittest.TestCase):
    def test_exp1_ulysses_gdn_kda_clc_triton_tilelang(self) -> None:
        exp_dir = ROOT / "experiments" / "flashinfer_ulysses_gdn_kda_clc_triton_tilelang_phase154"
        subprocess.check_call([sys.executable, str(exp_dir / "run.py"), "--check"])
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 154)
        self.assertGreater(data["ulysses_world8_gather"]["ulysses_speedup"], 1.4)
        self.assertEqual(data["gdn_kcp_recompute_handle"]["recompute_comm_handle_us"], 0.0)
        self.assertEqual(data["triton_and_tilelang_safety"]["capped_remat_iters"], 20)

    def test_exp2_sglang_lora_cp_vllm_nvfp4_engram_symmmem(self) -> None:
        exp_dir = ROOT / "experiments" / "sglang_lora_cp_vllm_nvfp4_engram_symmmem_phase154"
        subprocess.check_call([sys.executable, str(exp_dir / "run.py"), "--check"])
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["dsv41_engram_numa_cumem"]["host_ram_reduction_pct"], 75.0)
        self.assertGreater(data["dsv4_eplb_symm_mem_migration"]["eplb_migration_speedup"], 3.0)
        self.assertEqual(data["disagg_lease_and_envelope"]["long_queue_lease_expiry_aborts_after"], 0)

    def test_exp3_megatron_mla_nvls_symmmem_mooncake_nixl_redis(self) -> None:
        exp_dir = ROOT / "experiments" / "megatron_mla_nvls_symmmem_mooncake_nixl_redis_phase154"
        subprocess.check_call([sys.executable, str(exp_dir / "run.py"), "--check"])
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["megatron_mla_fused_rope_quant"]["mla_hbm_savings_pct"], 40.0)
        self.assertGreater(data["lmcache_and_nixl_redis"]["lmcache_nixl_speedup"], 4.0)
        self.assertEqual(data["mooncake_dfs_and_pytorch_compiler"]["dfs_recovery_rate_pct"], 98.0)


if __name__ == "__main__":
    unittest.main()
