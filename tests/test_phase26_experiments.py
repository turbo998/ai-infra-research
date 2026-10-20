"""Unit tests for Phase 26 deterministic experiments."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashmla_sm100_v41_fused_dequant_sim.run import (
    build_summary as build_flashmla_summary,
)
from experiments.lmcache_devdax_apc_dynamo_relay_sim.run import (
    build_summary as build_lmcache_dynamo_summary,
)
from experiments.sglang_kvshard_uncached_bucket_sim.run import (
    build_summary as build_sglang_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase26Experiments(unittest.TestCase):
    def test_flashmla_sm100_v41_fused_dequant_sim(self) -> None:
        summary = build_flashmla_summary()
        self.assertEqual(summary["kv_cache_formats"]["V32"]["bytes_per_token"], 656)
        self.assertEqual(summary["kv_cache_formats"]["V4"]["bytes_per_token"], 584)
        self.assertEqual(summary["kv_cache_formats"]["V41"]["bytes_per_token"], 528)
        self.assertEqual(summary["kv_cache_formats"]["V41_FP4"]["bytes_per_token"], 288)
        self.assertEqual(summary["dequant_banks"]["fp4_1cta_padded_288b_lds32_way"], 1)
        self.assertTrue(summary["e2m1_e4m3_bf16_exactness"]["all_products_exact_in_bf16"])
        self.assertEqual(summary["fused_sm100"]["tmem_layouts"]["hq_64"]["total_tmem_cols"], 512)
        self.assertEqual(summary["fused_sm100"]["tmem_layouts"]["hq_128"]["total_tmem_cols"], 480)
        res = subprocess.run(
            [sys.executable, "experiments/flashmla_sm100_v41_fused_dequant_sim/run.py", "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_sglang_kvshard_uncached_bucket_sim(self) -> None:
        summary = build_sglang_summary()
        self.assertEqual(summary["kvshard_pd"]["max_bootstrap_input_len"], 65_216)
        self.assertEqual(summary["kvshard_pd"]["skipped_empty_non_final_chunks"], 3)
        self.assertTrue(summary["kvshard_pd"]["lossless_page_reconstruction"])
        self.assertTrue(summary["router_and_eplb"]["eplb_qwen3_moe"]["exact_with_dispatch_info"])
        res = subprocess.run(
            [sys.executable, "experiments/sglang_kvshard_uncached_bucket_sim/run.py", "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_lmcache_devdax_apc_dynamo_relay_sim(self) -> None:
        summary = build_lmcache_dynamo_summary()
        self.assertTrue(summary["lmcache"]["dax_probes"]["second_256gib_disjoint"]["admitted"])
        self.assertFalse(
            summary["lmcache"]["dax_probes"]["overlapping_range_128gib_to_384gib"]["admitted"]
        )
        self.assertEqual(summary["lmcache"]["apc_attribution"]["patched_miss_l1_hit_rate"], 0.0)
        self.assertEqual(summary["dynamo"]["cuckoo_dc_replacement"]["zombie_candidates_pruned"], 2)
        self.assertEqual(
            summary["dynamo"]["migration_usage_rebase"]["rebased_client_usage"]["prompt_tokens"], 3
        )
        res = subprocess.run(
            [sys.executable, "experiments/lmcache_devdax_apc_dynamo_relay_sim/run.py", "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
