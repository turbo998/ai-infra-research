"""Unit tests for Phase 66 (Priority 75) experiments across Track A, Track B, and Track C."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPhase66Experiments(unittest.TestCase):
    def test_track_a_flashinfer_cudnn_prefill_splitk_fence_tilelang(self) -> None:
        mod = load_module(
            "experiments/flashinfer-cudnn-prefill-splitk-fence-tilelang-nvrtc/run.py"
        )
        res = mod.run_experiment()
        self.assertEqual(res["phase"], 66)
        self.assertEqual(res["priority"], 75)

        prefill = res["cudnn_prefill_plan_reuse"]
        self.assertAlmostEqual(prefill["max_plan_reduction_pct"], 14.93, places=2)
        self.assertAlmostEqual(prefill["max_plan_plus_run_reduction_pct"], 9.80, places=2)
        self.assertTrue(
            prefill["transitions"]["retained_metadata_inplace_as_strided"][
                "recomputes_descriptor_key"
            ]
        )

        splitk = res["splitk_slab_async_proxy_fence"]
        self.assertEqual(splitk["total_build_entries"], 247)
        self.assertEqual(splitk["arch_build_entries"], {"sm_100a": 80, "sm_103a": 80, "sm_107a": 87})
        self.assertEqual(splitk["refreshed_kernels"]["split_k_slab_sb2_fenced"], 6)

        pruning = res["cake_wgrad_and_softmax_pruning"]
        self.assertEqual(pruning["wgrad_folded_clones"], 3)
        self.assertEqual(pruning["combined_net_lines_removed"], 4954)

        tl = res["tilelang_nvrtc_and_vectorize_clone"]
        self.assertTrue(tl["autotune_save_succeeds_on_disk_cache_hit"])
        self.assertTrue(tl["analyzer_clone_isolated"])

    def test_track_b_sglang_gguf_moe_trtllm_mxfp8_vllm_shared_rope(self) -> None:
        mod = load_module(
            "experiments/sglang-gguf-moe-trtllm-mxfp8-e8m0-vllm-shared-rope/run.py"
        )
        res = mod.run_experiment()
        self.assertEqual(res["phase"], 66)
        self.assertEqual(res["priority"], 75)
        self.assertTrue(all(res["invariants"].values()))

        mxfp8 = res["trtllm_mxfp8_e8m0_subnormal_nan"]
        self.assertTrue(mxfp8["is_fp32_subnormal"])
        self.assertTrue(mxfp8["legacy_zero_lane_product_is_nan"])
        self.assertTrue(mxfp8["fixed_exp2f_rcp_is_finite"])
        self.assertEqual(mxfp8["fixed_zero_lane_product"], 0.0)
        self.assertEqual(mxfp8["fixed_max_lane_product"], 448.0)

        sgl = res["sglang_gguf_moe_and_cosmos_drain"]
        self.assertEqual(sgl["legacy_tile_y_ds_populated_rows"], 8)
        self.assertEqual(sgl["fixed_tile_y_ds_populated_rows"], 64)
        self.assertTrue(sgl["mul_rn_f32_preserves_subnormal"])
        self.assertTrue(sgl["cosmos_dreams_drain_after_outputs_clear"])

        vllm = res["vllm_shared_rope_and_shutdown_grace"]
        self.assertEqual(vllm["legacy_rope_vram_mib"], 2048.0)
        self.assertEqual(vllm["fixed_rope_vram_mib"], 4.0)
        self.assertEqual(vllm["rope_vram_reduction_x"], 512.0)
        self.assertEqual(vllm["timeout_cases"]["cuda_custom_env_30s_abort"], 30.0)
        self.assertEqual(vllm["timeout_cases"]["cuda_custom_env_30s_after_drain_expired"], 0)

    def test_track_c_megatron_padding_te_fp8_lmcache_swa_mooncake_nixl(self) -> None:
        mod = load_module(
            "experiments/megatron-padding-zero-lmcache-swa-replay-mooncake-wrid/run.py"
        )
        res = mod.run_experiment()
        self.assertEqual(res["phase"], 66)
        self.assertEqual(res["priority"], 75)

        meg = res["megatron_padding_zero_and_quantile_balancing"]
        self.assertEqual(meg["device_ptr_kernel_zeroed_rows_on_replay"], 56)
        self.assertEqual(meg["sp_zeroed_rows_per_rank"], [0, 0, 24, 32])
        self.assertEqual(meg["fp8_precision_factor_recovered"], 32.0)

        te = res["te_compiled_fp8_weight_caching"]
        self.assertEqual(te["weight_cast_reduction_pct"], 87.5)

        lmc = res["lmcache_swa_prefix_replay_alignment"]
        self.assertEqual(lmc["legacy_swa_replay"]["buffer_overflow_tokens"], 4)
        self.assertEqual(lmc["fixed_swa_replay"]["buffer_overflow_tokens"], 0)
        self.assertEqual(lmc["fixed_swa_replay"]["released_lookup_lock_range"], [8, 12])

        fab = res["mooncake_wrid_and_nixl_treq"]
        self.assertTrue(fab["legacy_overwrites_inflight_slot_0"])
        self.assertFalse(fab["fixed_overwrites_inflight_slot_0"])
        self.assertEqual(fab["nixl_treq_wire_bytes"], 16)
        self.assertTrue(fab["nixl_treq_roundtrip_ok"])

    def test_cli_check_mode_and_phase66_benchmark_html(self) -> None:
        slugs = [
            "flashinfer-cudnn-prefill-splitk-fence-tilelang-nvrtc",
            "sglang-gguf-moe-trtllm-mxfp8-e8m0-vllm-shared-rope",
            "megatron-padding-zero-lmcache-swa-replay-mooncake-wrid",
        ]
        for slug in slugs:
            script = ROOT / "experiments" / slug / "run.py"
            summary = ROOT / "experiments" / slug / "summary.json"
            self.assertTrue(summary.exists(), f"Missing {summary}")
            proc = subprocess.run(
                [sys.executable, str(script), "--check"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            data = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(data["experiment"], slug)

        html_path = ROOT / "docs" / "reports" / "ai-infra-phase66-benchmark-cards.html"
        self.assertTrue(html_path.exists())
        html = html_path.read_text(encoding="utf-8")
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", html)
        for i in range(1, 8):
            self.assertIn(f'id="card-{i}"', html)


if __name__ == "__main__":
    unittest.main()
