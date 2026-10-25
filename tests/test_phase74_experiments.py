import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase74Experiments(unittest.TestCase):
    def test_track_a_flashinfer_sm103_qsa_mhc4096_sm120_piped_kimi_k3_tilelang(self) -> None:
        exp_dir = REPO_ROOT / "experiments" / "flashinfer_sm103_qsa_persistent_mhc4096_sm120_piped_kimi_k3_tilelang_sim"
        sys.path.insert(0, str(exp_dir))
        try:
            import run as mod  # type: ignore
            summary = mod.build_summary()
        finally:
            sys.path.pop(0)
            sys.modules.pop("run", None)

        self.assertEqual(summary["phase"], 74)
        self.assertEqual(summary["priority"], 83)
        inv = summary["key_invariants"]
        self.assertEqual(inv["sm103_rows128_walk_ctas"], 152)
        self.assertEqual(inv["sm103_rows128_walk_tiles_max"], 56)
        self.assertAlmostEqual(inv["sm103_rows128_speedup_x"], 1.0613, places=4)
        self.assertAlmostEqual(inv["sm120_rows128_hnd_cudagraph_speedup_x"], 1.9018, places=4)
        self.assertAlmostEqual(inv["mega_mhc_b200_t1024_speedup_x"], 1.426, places=3)
        self.assertAlmostEqual(inv["mega_mhc_gb300_t1024_speedup_x"], 1.387, places=3)
        self.assertAlmostEqual(inv["sm120_pcie_96mib_bf16_speedup_x"], 1.1109, places=4)
        self.assertAlmostEqual(inv["kimi_k3_front_t64_speedup_x"], 1.2012, places=4)
        self.assertTrue(inv["tilelang_barrier_collision_eliminated"])

        res = subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track_b_trtllm_dsv4_rubin_vllm_d2d_seed_backpressure_sglang_msa_nvfp4(self) -> None:
        exp_dir = REPO_ROOT / "experiments" / "trtllm_dsv4_rubin_vllm_d2d_seed_backpressure_sglang_msa_nvfp4_sim"
        sys.path.insert(0, str(exp_dir))
        try:
            import run as mod  # type: ignore
            summary = mod.build_summary()
        finally:
            sys.path.pop(0)
            sys.modules.pop("run", None)

        self.assertEqual(summary["phase"], 74)
        self.assertEqual(summary["priority"], 83)
        inv = summary["key_invariants"]
        self.assertAlmostEqual(inv["trtllm_moe_saved_ms_batch8"], 1.525, places=3)
        self.assertAlmostEqual(inv["vllm_d2d_warmup_speedup_x"], 46.25, places=2)
        self.assertAlmostEqual(inv["vllm_drop_prob_after_3_failures"], 1.0, places=4)
        self.assertEqual(inv["sglang_msa_v1_selected_tokens"], 2048)
        self.assertAlmostEqual(inv["sglang_msa_v1_bw_reduction_128k_x"], 156.04, places=2)
        self.assertAlmostEqual(inv["vr200_cost_per_1m_tokens_usd"], 0.09, places=2)

        res = subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track_c_mooncake_neuron_pci_efa_evict_lmcache_mlu_torchtitan_te_nixl_dynamo(self) -> None:
        exp_dir = REPO_ROOT / "experiments" / "mooncake_neuron_pci_efa_evict_lmcache_mlu_torchtitan_te_nixl_dynamo_sim"
        sys.path.insert(0, str(exp_dir))
        try:
            import run as mod  # type: ignore
            summary = mod.build_summary()
        finally:
            sys.path.pop(0)
            sys.modules.pop("run", None)

        self.assertEqual(summary["phase"], 74)
        self.assertEqual(summary["priority"], 83)
        inv = summary["key_invariants"]
        self.assertAlmostEqual(inv["mooncake_16dev_efa_gain_x"], 18.78, places=2)
        self.assertEqual(inv["mooncake_evicted_peer_map_entries"], 1)
        self.assertAlmostEqual(inv["lmcache_mlu_staging_reduction_x"], 15.74, places=2)
        self.assertEqual(inv["torchtitan_expert_fsdp_reduction_x"], 8)
        self.assertAlmostEqual(inv["torchtitan_qwen3_rollout_wait_speedup_x"], 1.1618, places=4)
        self.assertAlmostEqual(inv["nixl_arm_same_socket_gain_x"], 4.3613, places=4)

        res = subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
