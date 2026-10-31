import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase133Experiments(unittest.TestCase):
    def test_track_a_phase133(self) -> None:
        exp = ROOT / "experiments" / "flashinfer_cake_router_qkrope_minimax_qwen38_mhc_triton_tilelang_phase133"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        summary = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["priority"], 142)
        m = summary["metrics"]
        self.assertGreater(m["b200_1tok_router_speedup"], 1.23)
        self.assertGreater(m["gb300_prefill_persistent_speedup"], 1.23)
        self.assertGreater(m["minimax_h3_b200_2stream_speedup"], 1.20)
        self.assertEqual(m["qwen38_cudnn_vendored_files_after"], 160)
        self.assertGreater(m["qwen38_extend_speedup"], 3.68)
        self.assertEqual(m["mhc_late_claim_deadlock_rate_pct"], 0.0)
        self.assertEqual(m["triton_inactive_gather_row_err_after"], 0.0)
        self.assertEqual(m["triton_hip_aot_zero_grid_errors_after"], 0)
        self.assertGreater(m["ascend_deepgemm_mte1_speedup"], 1.19)
        self.assertEqual(m["ascend_pto_vmi_lanes"], 128)

    def test_track_b_phase133(self) -> None:
        exp = ROOT / "experiments" / "sglang_pd_async_dspark_npu_canary_vllm_dflash_pynv_trtllm_pp_nccl_phase133"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        summary = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["priority"], 142)
        m = summary["metrics"]
        self.assertEqual(m["pd_inline_host_syncs_after"], 0)
        self.assertGreater(m["pd_decode_step_speedup"], 1.24)
        self.assertEqual(m["gemma4_unified_saved_gib"], 16.8)
        self.assertEqual(m["dflash_saved_gb_per_rank"], 2.85)
        self.assertEqual(m["dflash_saved_gb_per_8gpu_node"], 22.80)
        self.assertEqual(m["pynv_caps_speedup"], 46.25)
        self.assertEqual(m["trtllm_pp_nccl_graph_reg_faults_after"], 0)

    def test_track_c_phase133(self) -> None:
        exp = ROOT / "experiments" / "megatron_fsdp_prefetch_moe_unpermute_rocm_flydsl_mooncake_lmcache_dynamo_nixl_phase133"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        summary = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["priority"], 142)
        m = summary["metrics"]
        self.assertGreater(m["mfsdp_prefetch_step_speedup"], 1.14)
        self.assertEqual(m["moe_unpermute_speedup"], 4.0)
        self.assertEqual(m["moe_probs_released_saved_gib_64layers"], 2.0)
        self.assertEqual(m["gfx950_ck_addmm_fail_rate_after_pct"], 0.0)
        self.assertEqual(m["gfx950_flydsl_dma_races_after"], 0)
        self.assertGreater(m["hp_tcp_p99_speedup"], 2.89)
        self.assertEqual(m["lmcache_phantom_staging_leak_mib_after"], 0.0)
        self.assertGreater(m["dynamo_queue_gate_speedup"], 4.30)
        self.assertEqual(m["nixl_mooncake_reconnect_success_pct_after"], 100.0)


if __name__ == "__main__":
    unittest.main()
