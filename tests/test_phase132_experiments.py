import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase132Experiments(unittest.TestCase):
    def test_track_a_phase132(self) -> None:
        exp = ROOT / "experiments" / "flashinfer_cake_dsa_kda_moe_sampling_warp_decode_mhc_triton_tilelang_phase132"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        summary = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["priority"], 141)
        m = summary["metrics"]
        self.assertEqual(m["radix_passes_fused_on_hit"], 7)
        self.assertGreater(m["dsa_z3cp_speedup"], 1.02)
        self.assertEqual(m["premature_reused_blocks_after"], 0)
        self.assertEqual(m["dgrad_down_max_weight_elems"], 12582912)
        self.assertGreaterEqual(m["b200_probed_die_split_speedup"], 1.037)
        self.assertGreater(m["sampling_gb300_graph_b8_speedup"], 1.027)
        self.assertEqual(m["warp_decode_pre_wait_max_claims"], 3)
        self.assertEqual(m["kimi_k3_gb300_fb_m512_launches_after"], 1)
        self.assertGreater(m["kimi_k3_gb300_fb_m512_speedup"], 1.17)
        self.assertEqual(m["amd_spurious_alias_stall_pct_removed"], 30.0)
        self.assertEqual(m["nvidia_tmem_cga_layout_mismatches_after"], 0)
        self.assertEqual(m["tilelang_warp_reduce_inst_reduction"], 5.0)

    def test_track_b_phase132(self) -> None:
        exp = ROOT / "experiments" / "sglang_rl_weights_ltx_dp_vllm_hisparse_mxfp4_ple_trtllm_adp_ssm_phase132"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        summary = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["priority"], 141)
        m = summary["metrics"]
        self.assertEqual(m["rl_check_weights_speedup"], 32.0)
        self.assertEqual(m["untracked_stale_prompt_kv_spans_after"], 0)
        self.assertEqual(m["hisparse_reserved_import_step_tokens"], 4)
        self.assertEqual(m["hisparse_host_page_overcommit_rate_after"], 0.0)
        self.assertEqual(m["kv_offload_out_of_order_events_after"], 0)
        self.assertEqual(m["k3_int4_convert_chunk_values"], 134217728)
        self.assertEqual(m["k3_chunk_peak_fp32_mib"], 512.0)
        self.assertEqual(m["tp8_expert_chunk_launches_after"], 8)
        self.assertGreaterEqual(m["gfx942_tp8_requant_speedup"], 4.85)
        self.assertEqual(m["qwen4exp_ple_resident_gib_disk_offload"], 0.0)
        self.assertEqual(m["adp_monitored_ranks_after"], 8)

    def test_track_c_phase132(self) -> None:
        exp = ROOT / "experiments" / "megatron_fsdp_recompute_csa_rope_mooncake_cc_iouring_lmcache_dynamo_nixl_phase132"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        summary = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["priority"], 141)
        m = summary["metrics"]
        self.assertGreater(m["mfsdp_bwd_step_speedup"], 1.16)
        self.assertEqual(m["csa_output_rope_saved_bytes_per_layer"], 268435456)
        self.assertEqual(m["csa_output_rope_saved_gib_64layers"], 16.0)
        self.assertLessEqual(m["gfx950_seqlen1_rel_err_after"], 0.001)
        self.assertEqual(m["rdma_retry_storm_reduction_pct"], 88.0)
        self.assertEqual(m["iouring_uaf_staging_writes_after"], 0)
        self.assertEqual(m["lmcache_unaligned_odirect_failure_rate_after"], 0.0)
        self.assertEqual(m["nixl_ep_peer_vram_leak_after_disconnect"], 0)


if __name__ == "__main__":
    unittest.main()
