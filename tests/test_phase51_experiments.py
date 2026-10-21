import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase51Experiments(unittest.TestCase):
    def _run_and_load(self, exp_dir_name: str) -> dict:
        exp_dir = REPO_ROOT / "experiments" / exp_dir_name
        script = exp_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=f"Failed {script}: {res.stderr}\n{res.stdout}")
        summary_path = exp_dir / "summary.json"
        self.assertTrue(summary_path.is_file())
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(data["all_invariants_passed"])
        return data

    def test_flashinfer_minimax_h3_sm120_fp8pv_sm110_xqa_sm107_ssd_deepgemm_deslop(self) -> None:
        data = self._run_and_load(
            "flashinfer-minimax-h3-sm120-fp8pv-sm110-xqa-sm107-ssd-deepgemm-deslop"
        )
        mm = data["minimax_h3_sm120_fp8pv"]
        self.assertEqual(mm["rtx5090_reported_two_session_geomean"], 1.3814)
        self.assertEqual(mm["rtx_pro_6000_reported_two_session_geomean"], 1.1876)
        self.assertEqual(
            mm["programs_512_tok"], ["KV_STATS", "QUANTIZE_MB3", "ATTENTION_SHORT"]
        )
        self.assertEqual(
            mm["programs_8192_tok_seg4097"], ["KV_STATS", "QUANTIZE", "ATTENTION_BASE"]
        )
        xqa = data["sm110_xqa_and_sm107_ssd"]
        self.assertEqual(xqa["sm110_total_exported_programs"], 43)
        self.assertEqual(xqa["sm110_auto_e4m3_256rows"], "pair")
        self.assertEqual(xqa["sm110_auto_e4m3_128rows"], "tmem")
        self.assertEqual(xqa["sm107_ssd_max_main_kernel_speedup"], 1.268)
        dg = data["deepgemm_package_and_deslop"]
        self.assertEqual(dg["total_programs"], 46)
        self.assertEqual(dg["total_arch_builds"], 89)
        self.assertEqual(dg["deslop_net_lines_removed"], 12408)

    def test_sglang_radix_tree_evict_actions_dsv41_vl_sim_trtllm_perf_floor_vllm_rust(self) -> None:
        data = self._run_and_load(
            "sglang-radix-tree-evict-actions-dsv41-vl-sim-trtllm-perf-floor-vllm-rust"
        )
        sg = data["sglang_radix_tree_and_dsv41"]
        self.assertTrue(sg["tripwire_on_undrained_action"])
        self.assertEqual(sg["scattered_tp_sharded"]["token_shard_rank"], 11)
        self.assertEqual(sg["scattered_tp_sharded"]["num_token_shards"], 16)
        self.assertFalse(sg["scattered_tp_sharded"]["padded_rows_masked_after_sim"])
        trt = data["trtllm_history_floor"]
        self.assertEqual(trt["floor_epoch_ms"], 1790856543000)
        self.assertTrue(trt["unfloored_triggers_false_regression"])
        self.assertFalse(trt["floored_triggers_false_regression"])
        vllm = data["vllm_pooling_xpu_rust"]
        self.assertEqual(vllm["mixed_batch_chunked"], "rejected_VLLMValidationError")
        self.assertEqual(
            vllm["rust_same_kind_override_keeps_aux"]["value"], "schema_from_rf"
        )
        self.assertEqual(
            vllm["rust_cross_kind_conflict"]["status"], "error_400_multiple_constraints"
        )

    def test_te_jax_mhc_triton_nccl_ep_import_megatron_vmm_lmcache_mp_dynamo_nixl_mooncake(self) -> None:
        data = self._run_and_load(
            "te-jax-mhc-triton-nccl-ep-import-megatron-vmm-lmcache-mp-dynamo-nixl-mooncake"
        )
        te = data["te_jax_mhc_and_ep_import"]
        self.assertEqual(te["n_mix_valid_cols"], 24)
        self.assertEqual(te["h_padded_dim"], 32)
        self.assertLess(te["sinkhorn_max_bistochastic_error"], 1e-6)
        self.assertEqual(te["prec_bf16_with_norm_use_tf32_false"], "tf32x3")
        meg = data["megatron_vmm_and_lmcache_mp"]
        self.assertEqual(
            meg["megatron_vmm_unprovisioned_imex"]["cuMemCreate_attempts"], 2
        )
        dnm = data["dynamo_nixl_mooncake"]
        self.assertEqual(dnm["nixl_final_execution_counter"], 3)
        self.assertTrue(dnm["mooncake_escaped_slice_rejected"])
        self.assertTrue(dnm["dynamo_v1beta2_roundtrip_exact"])


if __name__ == "__main__":
    unittest.main()
