import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPhase28Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=ROOT)
        summary_path = ROOT / rel_dir / "results" / "summary.json"
        self.assertTrue(summary_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_te_nvfp4_symm_ar_deepgemm_mhc_epoch(self) -> None:
        data = self._run_and_load("experiments/te_nvfp4_symm_ar_deepgemm_mhc_epoch")
        self.assertEqual(data["phase"], 28)
        self.assertEqual(data["priority"], 37)
        symm = data["symmetric_all_reduce_cache"]
        self.assertEqual(symm["exact_numel_cache"]["retained_numel"], 135168)
        self.assertEqual(symm["quarter_pow2_bucket_cache"]["retained_numel"], 49152)
        self.assertLessEqual(symm["quarter_pow2_bucket_cache"]["max_padding_ratio"], 0.25)
        self.assertTrue(symm["quarter_pow2_bucket_cache"]["cuda_graphs_valid_after_128_growth"])

        nvfp4 = data["nvfp4_dynamo_metadata"]
        self.assertEqual(nvfp4["fixed_flattened_columnwise_shape_2d"], [256, 64])
        self.assertEqual(nvfp4["fixed_fake_swizzle_mismatch_count"], 0)
        self.assertTrue(nvfp4["sm120_rht_matrix_forwarded"])

        dg = data["deepgemm_mega_mhc_epoch_barrier"]
        self.assertEqual(dg["k_split_barrier_epoch_offset"], 64)
        self.assertEqual(dg["k_split_barrier_exit_count_offset"], 72)
        self.assertEqual(dg["legacy_gridid_stale_replay_reads"], 7)
        self.assertEqual(dg["fixed_epoch_stale_replay_reads"], 0)
        self.assertEqual(dg["final_epoch"], 8)

    def test_mooncake_deferred_offload_dfs_align_hotcache(self) -> None:
        data = self._run_and_load("experiments/mooncake_deferred_offload_dfs_align_hotcache")
        self.assertEqual(data["phase"], 28)
        offload = data["deferred_offload_lifecycle"]
        self.assertEqual(offload["single_key_stream_21_items"]["naive_persisted_keys"], 0)
        edge = offload["multi_cycle_edge_trace"]
        self.assertEqual(edge["persisted_tasks"], ["k1@gen1", "k2@gen1", "k4@gen1", "k5@gen1"])
        self.assertEqual(edge["zombie_tasks_count"], 0)

        dfs = data["dfs_shard_allocator"]
        self.assertEqual(dfs["legacy_max_1mib_objects"], 56)
        self.assertEqual(dfs["fixed_max_1mib_objects"], 64)
        self.assertEqual(dfs["per_object_inflation_pct_removed"], 12.5)
        self.assertTrue(dfs["fixed_full_shard_64mib_succeeds"])
        self.assertEqual(dfs["unaligned_offsets_count"], 0)

        hc = data["hotcache_device_pointer_guard"]
        self.assertEqual(hc["legacy_host_memcpy_on_device_segfaults"], 3)
        self.assertEqual(hc["fixed_host_memcpy_on_device_segfaults"], 0)

    def test_megatron_payload_staging_verl_hdo_offload(self) -> None:
        data = self._run_and_load("experiments/megatron_payload_staging_verl_hdo_offload")
        self.assertEqual(data["phase"], 28)
        st = data["dynamic_engine_payload_staging"]
        self.assertEqual(st["sync_max_step_latency_ms"], 102.0)
        self.assertEqual(st["async_max_step_latency_ms"], 12.0)
        self.assertEqual(st["max_step_latency_speedup_x"], 8.5)
        self.assertEqual(st["retained_gpu_mib_with_host_copy"], 0.0)
        self.assertTrue(st["reset_blocked_when_staging_in_flight"])

        ckpt = data["idle_rerun_state_checkpoint_load"]
        self.assertFalse(ckpt["legacy_strict_load_succeeds"])
        self.assertTrue(ckpt["fixed_strict_load_succeeds"])
        self.assertTrue(ckpt["fixed_ignore_rerun_state"])

        hdo = data["verl_hdo_optimizer_offload"]
        self.assertTrue(hdo["legacy_next_step_fused_adam_crashes"])
        self.assertFalse(hdo["fixed_next_step_fused_adam_crashes"])


if __name__ == "__main__":
    unittest.main()
