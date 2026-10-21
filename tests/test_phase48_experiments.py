import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase48Experiments(unittest.TestCase):
    EXP_IDS = (
        "flashinfer-qwen35-frost-rubin-moe-ep-backends-sm107-cake-deepgemm-mla-fp8",
        "sglang-disagg-deferred-kv-tp-sync-radix-walk-trtllm-render-gemma4-vllm-offload",
        "lmcache-grpc-bulk-int-codec-hooks-dynamo-fpm-gil-trace-dgdr-nixl-te-jax-mhc-torchtitan",
    )

    def _run_and_load(self, exp_id: str) -> dict:
        exp_dir = ROOT / "experiments" / exp_id
        run_py = exp_dir / "run.py"
        self.assertTrue(run_py.exists(), f"Missing {run_py}")
        proc = subprocess.run(
            [sys.executable, str(run_py), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS]", proc.stdout)
        summary_path = exp_dir / "summary.json"
        self.assertTrue(summary_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_topic1_flashinfer_qwen35_frost_moe_ep_and_cake(self) -> None:
        data = self._run_and_load(self.EXP_IDS[0])
        self.assertEqual(data["phase"], 48)
        self.assertEqual(data["priority"], 57)
        qwen = data["qwen35_frost_rubin_block"]
        self.assertEqual(qwen["target_arch"], "sm_107")
        self.assertEqual(qwen["vendored_frost_files"], 103)
        self.assertFalse(qwen["cudnn_runtime_dependency"])
        self.assertEqual(qwen["Activation_HBM_Reduction_Ratio"], 10.2222)
        moe = data["moe_ep_alltoall_and_sm107"]
        self.assertEqual(moe["alltoall_payload_reduction_ratio"], 1.9394)
        self.assertTrue(moe["sm107_mercury_validator_fixed"])
        cake = data["cake_self_contained_packages"]
        self.assertEqual(cake["batched_gemm_variants"], 6)
        self.assertEqual(cake["fp8_gemm_variants"], 2)

    def test_topic2_sglang_trtllm_and_vllm(self) -> None:
        data = self._run_and_load(self.EXP_IDS[1])
        self.assertEqual(data["phase"], 48)
        self.assertEqual(data["priority"], 57)
        sgl = data["sglang_disagg_and_radix"]
        self.assertTrue(sgl["unsynced_ranks_diverged"])
        self.assertFalse(sgl["synced_ranks_diverged"])
        self.assertTrue(sgl["free_before_cache_action_invariant"])
        trt = data["trtllm_render_gemma4_perf"]
        self.assertEqual(trt["gemma4_max_num_heads_per_kv"], 8)
        self.assertEqual(trt["kimi_k3_opener_tokens_excluded"], 3)
        self.assertTrue(trt["passes_with_floor"])
        vllm = data["vllm_offload_and_pooling"]
        self.assertTrue(vllm["placeholder_survives_reset"])
        self.assertTrue(vllm["multimodal_chunked_embedding_rejected"])

    def test_topic3_lmcache_dynamo_nixl_te_and_torchtitan(self) -> None:
        data = self._run_and_load(self.EXP_IDS[2])
        self.assertEqual(data["phase"], 48)
        self.assertEqual(data["priority"], 57)
        lmc = data["lmcache_codec_and_hooks"]
        self.assertEqual(lmc["new_python_calls_per_rpc"], 1)
        self.assertEqual(lmc["mp_plugin_hook_count"], 5)
        dyn = data["dynamo_and_nixl"]
        self.assertTrue(dyn["child_observes_parent_cancellation"])
        self.assertTrue(dyn["dgdr_v1beta2_mutually_exclusive_valid"])
        self.assertTrue(dyn["nixl_slash_in_source_id_rejected"])
        tet = data["te_jax_mhc_and_torchtitan"]
        self.assertEqual(tet["te_jax_mhc_n_mix"], 24)
        self.assertEqual(tet["torchtitan_saved_16k_mib"], 128.0)
        self.assertEqual(tet["torchtitan_saved_128k_mib"], 1043.983)


if __name__ == "__main__":
    unittest.main()
