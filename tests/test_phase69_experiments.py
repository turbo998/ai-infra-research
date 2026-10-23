"""Unit tests for Phase 69 (priority: 78) deterministic experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(rel_path: str, module_name: str):
    full_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPhase69Experiments(unittest.TestCase):
    def test_track_a_rubin_gdn_ugpu_kcp_tilelang(self) -> None:
        mod = _load_module(
            "experiments/flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1/run.py",
            "phase69_track_a",
        )
        summary = mod.build_summary()
        self.assertEqual(summary["phase"], 69)
        self.assertEqual(summary["priority"], 78)
        self.assertTrue(all(summary["invariants"].values()))

        # Check Rubin GDN uGPU metrics
        rubin = summary["rubin_gdn_ugpu"]
        self.assertEqual(rubin["pf_dist_tiles"], 1766)
        self.assertEqual(rubin["vr200_benchmarks"][0]["rubin_ugpu_speedup"], 1.46)
        self.assertEqual(rubin["vr200_benchmarks"][4]["rubin_ugpu_bw_pct"], 88)

        # Check Rubin MLA PDL switch and KCP wave_fill1
        mla = summary["rubin_mla_pdl_and_kcp"]
        self.assertTrue(mla["pdl_cases"][0]["enable_pdl"])
        self.assertFalse(mla["pdl_cases"][1]["enable_pdl"])
        self.assertEqual(mla["kcp_benchmarks"]["plan_1frag_nc64_h16"]["p_segments"], 9)

        # Check TileLang guards
        tl = summary["tilelang_guards"]
        self.assertTrue(tl["nd2nz_subtile_16x64_fp16"]["valid"])
        self.assertEqual(tl["nd2nz_subtile_16x64_fp16"]["dst_span_bytes"], 6656)
        self.assertEqual(tl["pto_float32x4_2slots"]["scalar_buffer_extent"], 8)
        self.assertFalse(tl["cuda_cast_cases"]["f32_to_e8m0_rs_rejected"])

    def test_track_b_vllm_topk_nan_aux_output_sglang_trtllm(self) -> None:
        mod = _load_module(
            "experiments/vllm-topk-nan-aux-output-sglang-inkling-trtllm/run.py",
            "phase69_track_b",
        )
        summary = mod.build_summary()
        self.assertEqual(summary["phase"], 69)
        self.assertEqual(summary["priority"], 78)
        self.assertTrue(all(summary["invariants"].values()))

        # Check NaN ordering
        topk = summary["vllm_topk_nan"]
        self.assertEqual(topk["rows"][-1]["buggy_key_hex"], "0x7FFFFFFF")
        self.assertEqual(topk["rows"][-1]["fixed_key_hex"], "0xFFC00000")

        # Check SGLang Inkling effort validation
        sgl = mod.simulate_sglang_inkling_and_trtllm_request()
        parse_fn = sgl["parse_inkling_effort_fn"]
        self.assertEqual(parse_fn("high"), 0.9)
        self.assertEqual(parse_fn("max"), 0.99)
        with self.assertRaises(ValueError):
            parse_fn(1.0)
        with self.assertRaises(ValueError):
            parse_fn(True)

    def test_track_c_te_mxfp8_rmsnorm_tma_torchtitan_lmcache_mooncake(self) -> None:
        mod = _load_module(
            "experiments/te-mxfp8-rmsnorm-tma-torchtitan-lmcache-mooncake/run.py",
            "phase69_track_c",
        )
        summary = mod.build_summary()
        self.assertEqual(summary["phase"], 69)
        self.assertEqual(summary["priority"], 78)
        self.assertTrue(all(summary["invariants"].values()))

        # Check TE swizzle and TMA wait depth
        te = summary["te_rmsnorm_mxfp8_and_tma"]
        self.assertTrue(te["swizzled_scale_bijective"])
        self.assertEqual(te["tma_wait_group_read_depth"], 2)

        # Check LMCache trace-analyze and Mooncake DFS-only
        lmc = summary["lmcache_trace_and_mooncake_dfs"]
        self.assertEqual(lmc["ordered_request_ids"], ["sess-r1", "sess-r2", "sess-r3", "sess-r4"])
        self.assertEqual(lmc["grid"][0]["prefix_hit_tokens"], 8)
        self.assertEqual(lmc["grid"][1]["prefix_hit_tokens"], 12)
        self.assertTrue(lmc["mooncake_dfs_only_valid"])

    def test_cli_check_mode_and_summary_files(self) -> None:
        exp_dirs = [
            "experiments/flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1",
            "experiments/vllm-topk-nan-aux-output-sglang-inkling-trtllm",
            "experiments/te-mxfp8-rmsnorm-tma-torchtitan-lmcache-mooncake",
        ]
        for rel_dir in exp_dirs:
            script = REPO_ROOT / rel_dir / "run.py"
            res = subprocess.run(
                [sys.executable, str(script), "--check"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("OK:", res.stdout)
            summary_path = REPO_ROOT / rel_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(data["evidence_tier"], "own-offline")
            self.assertEqual(data["phase"], 69)


if __name__ == "__main__":
    unittest.main()
