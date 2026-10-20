"""Unit tests for Phase 32 deterministic experiments and bilingual deliverables."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashinfer_cake_rmsnorm_sglang_ep_vllm_mrv2.run import (
    run_experiment as run_track1,
)
from experiments.mooncake_ha_drain_nixl_eventfd_lmcache_kimi.run import (
    run_experiment as run_track3,
)
from experiments.torchtitan_cuspy_graph_muse_rope_te_ep.run import (
    run_experiment as run_track2,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase32Experiments(unittest.TestCase):
    def test_track1_flashinfer_cake_rmsnorm_sglang_ep_vllm_mrv2(self) -> None:
        summary = run_track1()
        self.assertEqual(summary["phase"], 32)
        self.assertEqual(summary["priority"], 41)
        cake = summary["cake_rmsnorm_grad"]
        self.assertEqual(cake["routed_coop_ring_cases"], 7)
        self.assertEqual(cake["b200_h1536_t65536_speedup_x"], 1.1245)
        self.assertEqual(cake["r200_h4096_t65536_speedup_x"], 1.2161)
        self.assertEqual(cake["deterministic_dw_drift"], 0.0)

        ep = summary["sglang_gfx1250_w4a8_ep"]
        self.assertEqual(ep["expert_map_len"], 17)
        self.assertEqual(ep["oob_gather_without_clamp"], 72)
        self.assertEqual(ep["oob_gather_with_clamp"], 0)
        self.assertEqual(ep["max_ep_allreduce_abs_error"], 0.0)

        mrv2 = summary["vllm_mrv2_spec_async"]
        self.assertEqual(mrv2["mrv2_async_preserved_cases"], 3)
        self.assertEqual(mrv2["v1_incompat_blocked_cases"], 3)
        self.assertEqual(mrv2["rocm_dbo_blocked_cases"], 1)

    def test_track2_torchtitan_cuspy_graph_muse_rope_te_ep(self) -> None:
        summary = run_track2()
        cuspy = summary["cuspy_cupti_graph"]
        self.assertEqual(cuspy["edge_walk_captured_nodes"], 20)
        self.assertEqual(cuspy["cupti_callback_captured_nodes"], 24)
        self.assertEqual(cuspy["exec_key_nccl_event_nodes_resolved"], 4)
        self.assertTrue(cuspy["py_stack_64bit_key_bijective"])

        muse = summary["muse_spark_qk_rope"]
        self.assertEqual(muse["decode_256seqs_speedup_x"], 3.75)
        self.assertEqual(muse["decode_1seq_speedup_x"], 2.6905)
        self.assertEqual(muse["max_bitwise_diff"], 0.0)
        self.assertLessEqual(muse["compiled_graphs"], muse["recompile_limit"])

        te_ep = summary["te_jax_fused_nvl_ep"]
        self.assertEqual(te_ep["ffi_call_reduction_pct"], 50.0)
        self.assertLess(te_ep["max_combine_abs_diff"], 1e-12)

    def test_track3_mooncake_ha_drain_nixl_eventfd_lmcache_kimi(self) -> None:
        summary = run_track3()
        ha = summary["mooncake_ha_sigterm_drain"]
        self.assertEqual(ha["ordered_drain_stage_count"], 9)
        self.assertTrue(ha["normal_drain_matches_order"])
        self.assertEqual(ha["drained_unflushed_oplog_batches"], 0)
        self.assertTrue(ha["watchdog_fired_on_hang"])
        self.assertEqual(ha["second_sigterm_exit_code"], 143)

        nixl = summary["nixl_ucx_eventfd"]
        self.assertEqual(nixl["fd_reduction_pct"], 50.0)
        self.assertEqual(nixl["fds_saved_at_256_threads"], 256)

        lmcache = summary["lmcache_kimi_linear_mamba"]
        self.assertEqual(lmcache["legacy_accepted_states"], 1)
        self.assertEqual(lmcache["new_accepted_states"], 2)
        self.assertEqual(lmcache["non_contiguous_rejected"], 1)
        self.assertEqual(lmcache["kimi_conv_wire_shape"], [5, 1, 1, 21])

    def test_cli_check_and_bilingual_deliverables(self) -> None:
        scripts = [
            "experiments/flashinfer_cake_rmsnorm_sglang_ep_vllm_mrv2/run.py",
            "experiments/torchtitan_cuspy_graph_muse_rope_te_ep/run.py",
            "experiments/mooncake_ha_drain_nixl_eventfd_lmcache_kimi/run.py",
        ]
        for rel in scripts:
            res = subprocess.run(
                [sys.executable, str(REPO_ROOT / rel), "--check"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertIn("[PASS]", res.stdout)

        docs = [
            "docs/components/flashinfer-cake-rmsnorm-grad-sglang-gfx1250-mxfp4-ep-vllm-mrv2-spec-async.md",
            "docs/components/flashinfer-cake-rmsnorm-grad-sglang-gfx1250-mxfp4-ep-vllm-mrv2-spec-async.en.md",
            "docs/components/torchtitan-cuspy-cupti-graph-muse-spark-rope-norm-te-nvl-fused-ep.md",
            "docs/components/torchtitan-cuspy-cupti-graph-muse-spark-rope-norm-te-nvl-fused-ep.en.md",
            "docs/components/mooncake-ha-sigterm-drain-nixl-ucx-eventfd-lmcache-kimi-linear.md",
            "docs/components/mooncake-ha-sigterm-drain-nixl-ucx-eventfd-lmcache-kimi-linear.en.md",
            "docs/reports/ai-infra-phase32-benchmark-cards.html",
        ]
        for d in docs:
            path = REPO_ROOT / d
            self.assertTrue(path.exists(), f"Missing {d}")

        report_html = (REPO_ROOT / "docs/reports/ai-infra-phase32-benchmark-cards.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", report_html)
        for step in ("1 / 7", "2 / 7", "3 / 7", "4 / 7", "5 / 7", "6 / 7", "7 / 7"):
            self.assertIn(step, report_html)


if __name__ == "__main__":
    unittest.main()
