"""Unit tests for Phase 31 (priority 40) deterministic experiments and bilingual deliverables."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.mooncake_gid_recovery_lmcache_inproc_nixl_ucx.run import (
    run_experiment as run_mooncake_lmcache_nixl,
)
from experiments.sglang_dsv4_mhc_flashinfer_gdn_trtllm_megamoe.run import (
    run_experiment as run_sglang_flashinfer_trtllm,
)
from experiments.torchtitan_mla_compile_te_mxfp8_verl_tail_kl.run import (
    run_experiment as run_torchtitan_te_verl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase31Experiments(unittest.TestCase):
    def test_sglang_dsv4_mhc_flashinfer_gdn_trtllm_megamoe(self) -> None:
        summary = run_sglang_flashinfer_trtllm()
        self.assertTrue(summary["invariants_verified"])
        sgl = summary["sglang_dsv4_dspark_mhc"]
        self.assertEqual(sgl["legacy_sm100_tf32_mhc_hits"], 2)
        self.assertEqual(sgl["patched_sm100_tf32_mhc_hits"], 6)
        self.assertGreater(sgl["dspark_mhc_speedup_ratio"], 1.9)
        self.assertEqual(sgl["stale_cache_stages_with_post_hook"], 0)

        fi = summary["flashinfer_gdn_tmem_fence"]
        self.assertEqual(fi["legacy_tmem_waw_hazards"], 16)
        self.assertEqual(fi["fenced_tmem_waw_hazards"], 0)

        trt = summary["trtllm_megamoe_tep_contract"]
        self.assertTrue(trt["double_allreduce_eliminated"])
        self.assertEqual(trt["opt_in_abs_error"], 0.0)
        self.assertEqual(trt["generic_abs_error"], 0.0)

    def test_torchtitan_mla_compile_te_mxfp8_verl_tail_kl(self) -> None:
        summary = run_torchtitan_te_verl()
        self.assertTrue(summary["invariants_verified"])
        mla = summary["torchtitan_mla_qk_local_compile"]
        self.assertEqual(mla["patched_compile_masked_load_branches"], 0)
        self.assertGreater(mla["speedup_vs_default_compile"], 1.7)

        te = summary["te_blackwell_fp8bs_mxfp8"]
        self.assertEqual(te["max_pow2_broadcast_error"], 0.0)
        self.assertEqual(te["non_pow2_scales_rejected_on_blackwell"], 4)
        self.assertTrue(te["valid_128_aligned_splits_use_grouped_tensor"])
        self.assertTrue(te["unaligned_splits_fallback_triggered"])

        verl = summary["verl_topk_tail_kl"]
        self.assertLess(verl["truncated_forward_kl_topk"], 0.0)
        self.assertGreater(verl["tail_aware_forward_kl_total"], 0.4)
        self.assertTrue(verl["clamp_breaks_simplex_guard_raised"])

    def test_mooncake_gid_recovery_lmcache_inproc_nixl_ucx(self) -> None:
        summary = run_mooncake_lmcache_nixl()
        self.assertTrue(summary["invariants_verified"])
        mc = summary["mooncake_auto_gid_recovery"]
        self.assertEqual(mc["patched_epoch_advances"], 1)
        self.assertEqual(mc["patched_duplicate_epoch_completions_ignored"], 15)
        self.assertEqual(mc["patched_retried_slices_succeeded"], 16)
        self.assertFalse(mc["patched_rail_blacklisted"])

        lmc = summary["lmcache_inprocess_isolation"]
        self.assertEqual(lmc["invalid_configs_caught_by_value_error"], 3)
        self.assertEqual(lmc["deep_stream_crashes_prevented"], 3)

        nixl = summary["nixl_ucx_phase_tracing"]
        self.assertTrue(nixl["wire_completed_omitted_to_prevent_poll_skew"])
        self.assertGreater(nixl["poll_skew_inflation_ratio"], 8.0)

    def test_cli_check_flags_and_bilingual_docs(self) -> None:
        scripts = [
            REPO_ROOT / "experiments" / "sglang_dsv4_mhc_flashinfer_gdn_trtllm_megamoe" / "run.py",
            REPO_ROOT / "experiments" / "torchtitan_mla_compile_te_mxfp8_verl_tail_kl" / "run.py",
            REPO_ROOT / "experiments" / "mooncake_gid_recovery_lmcache_inproc_nixl_ucx" / "run.py",
        ]
        for script in scripts:
            proc = subprocess.run(
                [sys.executable, str(script), "--check"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

        docs = [
            "docs/components/sglang-dsv4-dspark-tf32-mhc-flashinfer-gdn-tmem-trtllm-megamoe.md",
            "docs/components/sglang-dsv4-dspark-tf32-mhc-flashinfer-gdn-tmem-trtllm-megamoe.en.md",
            "docs/components/torchtitan-dsv3-mla-local-compile-te-mxfp8-verl-topk-kl-rollout.md",
            "docs/components/torchtitan-dsv3-mla-local-compile-te-mxfp8-verl-topk-kl-rollout.en.md",
            "docs/components/mooncake-auto-gid-recovery-reshard-inflight-lmcache-nixl-ucx-spans.md",
            "docs/components/mooncake-auto-gid-recovery-reshard-inflight-lmcache-nixl-ucx-spans.en.md",
            "docs/reports/ai-infra-phase31-benchmark-cards.html",
        ]
        for rel in docs:
            p = REPO_ROOT / rel
            self.assertTrue(p.exists(), f"Missing deliverable: {rel}")


if __name__ == "__main__":
    unittest.main()
