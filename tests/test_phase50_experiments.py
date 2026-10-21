"""Unit tests for Phase 50 deterministic stdlib experiments."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase50Experiments(unittest.TestCase):
    def _run_experiment(self, slug: str) -> dict:
        script = ROOT / "experiments" / slug / "run.py"
        self.assertTrue(script.is_file(), f"Missing script: {script}")
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        summary_path = ROOT / "experiments" / slug / "summary.json"
        self.assertTrue(summary_path.is_file(), f"Missing summary.json: {summary_path}")
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment"], slug)
        self.assertIn(slug, proc.stdout)
        return data

    def test_flashinfer_trtllm_gen_sliding_varwindow_and_cake_consolidation(self) -> None:
        slug = "flashinfer-trtllm-gen-sliding-varwindow-fmha-cake-minimax-h3-small-dense-gemm"
        data = self._run_experiment(slug)
        fmha = data["trtllm_gen_fmha"]
        self.assertEqual(fmha["total_fmha_cubins"], 36964)
        self.assertEqual(fmha["variable_window_cubins"], 1424)
        self.assertFalse(fmha["decode_non_binding"]["fixed_window_binds"])
        self.assertEqual(fmha["decode_non_binding"]["mask_type"], 1)  # CAUSAL fast path
        self.assertTrue(fmha["decode_binding"]["fixed_window_binds"])
        self.assertEqual(fmha["decode_binding"]["mask_type"], 2)  # SLIDING_OR_CHUNKED_CAUSAL
        self.assertEqual(fmha["var_window"]["mask_type"], 5)  # VARIABLE_WINDOW
        self.assertEqual(fmha["chunked"]["chunked_attention_size_log2"], 6)
        self.assertEqual(fmha["prefill_b300_overhead_pct"], 4.17)
        self.assertEqual(fmha["prefill_b200_overhead_pct"], 3.69)

        cake = data["cake_consolidation"]
        self.assertEqual(cake["total_lines_added"], 6238)
        self.assertEqual(cake["total_lines_deleted"], 45536)
        self.assertEqual(cake["total_net_lines_removed"], 39298)
        self.assertEqual(cake["dense_projection"]["exported_templates_union"], 84)

    def test_sglang_mamba_zero_sync_kda_reuseport_trtllm_indexer_vllm_rust(self) -> None:
        slug = "sglang-mamba-zero-sync-kda-graph-reuseport-trtllm-indexer-prewarm-vllm-rust"
        data = self._run_experiment(slug)
        mamba = data["sglang_mamba_zero_sync"]
        self.assertEqual(mamba["gathered_slots"], [10, 31, 50])
        self.assertEqual(mamba["device_to_host_sync_count"], 0)
        self.assertFalse(mamba["isin_sync_at_n4096_p20"])
        self.assertTrue(mamba["isin_sync_at_n4096_p34"])
        self.assertEqual(mamba["searchsorted_multimodal_mask"], [False, True, False, True, False, False])

        kda = data["sglang_kda_and_reuseport"]
        self.assertTrue(kda["contains_last_prefill_chunk_full"])
        self.assertFalse(kda["contains_last_prefill_chunk_tbo_split0"])
        self.assertTrue(kda["contains_last_prefill_chunk_tbo_split1"])
        self.assertEqual(kda["reuseport_plan"]["total_bound_sockets"], 9)

        trt = data["trtllm_indexer_prewarm"]
        self.assertEqual(trt["local_indexer_layers"], [1, 2])
        self.assertEqual(trt["prewarm_token_counts"], [4, 8, 16])
        self.assertEqual(trt["total_prewarm_invocations"], 6)

        vllm = data["vllm_cosmos3_and_rust"]
        self.assertEqual(len(vllm["remapped_keys"]), 3)
        self.assertEqual(vllm["rust_prompt_fills_context"]["error_variant"], "PromptFillsContext")
        self.assertEqual(vllm["rust_prompt_too_long"]["error_variant"], "PromptTooLong")

    def test_te_kda_torchtitan_cuspy_kimi_eps_lmcache_dynamo_mooncake_tent(self) -> None:
        slug = "te-kda-attention-torchtitan-cuspy-kimi-eps-lmcache-inprocess-mooncake-tent"
        data = self._run_experiment(slug)
        te = data["transformer_engine"]
        self.assertGreater(te["kda_vector_gate_noncomm_max_diff"], 0.04)
        self.assertEqual(te["kda_scalar_gate_comm_max_diff"], 0.0)
        self.assertTrue(te["thd_learnable_softmax_cudnn925_d128"])
        self.assertFalse(te["thd_learnable_softmax_cudnn925_d96"])

        tt = data["torchtitan"]
        self.assertEqual(tt["kimi_rms_norm_eps"], 1e-5)
        self.assertGreaterEqual(tt["low_norm_token_drift_pct"], 30.0)

        mc = data["mooncake"]
        self.assertEqual(mc["exit_code_pass"], 0)
        self.assertEqual(mc["exit_code_check_fail"], 1)
        self.assertEqual(mc["exit_code_collector_fail"], 3)
        self.assertEqual(mc["exit_code_internal_fail"], 4)
        self.assertEqual(mc["int_ragged_roundtrip"]["fixed_roundtrip"][2][0], (1 << 53) + 1)
        self.assertEqual(mc["bool_ragged_roundtrip"]["fixed_codec"], "msgpack_ragged")


if __name__ == "__main__":
    unittest.main()
