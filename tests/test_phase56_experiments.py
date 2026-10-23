import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPhase56Experiments(unittest.TestCase):
    def _run_and_load(self, slug: str) -> dict:
        exp_dir = ROOT / "experiments" / slug
        script = exp_dir / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        return json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))

    def test_flashinfer_vsa_sm90_sage_fp8_kda_kcp_dsa_holefree_conv1d_gdp(self) -> None:
        data = self._run_and_load(
            "flashinfer-vsa-sm90-sage-fp8-kda-kcp-dsa-holefree-conv1d-gdp"
        )
        self.assertTrue(data["all_invariants_passed"])
        self.assertEqual(data["cake_vsa_sm90_sage_fp8"]["total_stages"], 40)
        self.assertEqual(data["cake_vsa_sm90_sage_fp8"]["suite_row_total"], 137)
        self.assertEqual(data["cake_kcp_and_lean_arena"]["kcp_programs_after"], 8)
        self.assertEqual(data["cake_kcp_and_lean_arena"]["chosen_p_balanced_2x64"], 1)
        self.assertEqual(data["cake_kcp_and_lean_arena"]["lean_arena_reduction_pct"], 37.17)
        self.assertEqual(data["dsa_hole_compaction_and_conv1d"]["dsa_gb300_speedup"], 2.247)
        self.assertEqual(len(data["dsa_hole_compaction_and_conv1d"]["conv1d_routes"]), 6)

    def test_sglang_hicache_mla_dcp_diffusion_lora_trtllm_swa_rubin_vllm_mxfp4(self) -> None:
        data = self._run_and_load(
            "sglang-hicache-mla-dcp-diffusion-lora-trtllm-swa-rubin-vllm-mxfp4"
        )
        self.assertTrue(data["all_invariants_passed"])
        self.assertEqual(
            data["sglang_hicache_mla_dcp_and_diffusion"]["active_writer_ranks"], [0, 1]
        )
        self.assertEqual(
            data["sglang_hicache_mla_dcp_and_diffusion"]["best_agreed_checkpoint_page"], 4
        )
        self.assertEqual(
            data["trtllm_swa_lowm_rubin_and_v_stride"]["swa_snapshot_point_div175_blk32"], 160
        )
        self.assertIsNone(
            data["trtllm_swa_lowm_rubin_and_v_stride"]["swa_snapshot_point_after_reuse"]
        )
        self.assertEqual(
            data["vllm_mxfp4_and_rust_frontend"]["lora_msgpack_fields_after"], 6
        )

    def test_megatron_gdp_prefill_a2a_recompute_te_mask_titan_rl_lmcache_mooncake_nixl_dynamo(
        self,
    ) -> None:
        data = self._run_and_load(
            "megatron-gdp-prefill-a2a-recompute-te-mask-titan-rl-lmcache-mooncake-nixl-dynamo"
        )
        self.assertTrue(data["all_invariants_passed"])
        self.assertEqual(
            len(data["megatron_te_and_torchtitan"]["megatron_mutable_chunk_state_fields"]), 5
        )
        self.assertEqual(data["megatron_te_and_torchtitan"]["te_kept_count"], 5)
        self.assertEqual(data["megatron_te_and_torchtitan"]["torchtitan_new_trainer_gpus"], 128)
        self.assertEqual(
            data["lmcache_mooncake_nixl_dynamo"]["lmcache_swa_aligned_need"], 8
        )
        self.assertEqual(
            data["lmcache_mooncake_nixl_dynamo"]["mooncake_fixed_retry_slot_on_fail"], 255
        )
        self.assertEqual(
            data["lmcache_mooncake_nixl_dynamo"]["dynamo_edge_slot_after_bytes"], 33
        )


if __name__ == "__main__":
    unittest.main()
