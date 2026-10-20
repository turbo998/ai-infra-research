from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashinfer_rubin_moe_barrier_ep_split_ag_matmul.run import (
    run_all as run_flashinfer,
)
from experiments.lmcache_dax_cxl_offset_grpc_mooncake_oplog.run import (
    run_all as run_lmcache_mooncake,
)
from experiments.torchtitan_dsv3_mla_rope_megatron_recompute.run import (
    run_all as run_torchtitan_megatron,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase30Experiments(unittest.TestCase):
    def test_flashinfer_rubin_moe_barrier_ep_split_ag_matmul(self) -> None:
        summary = run_flashinfer()
        rubin = summary["rubin_finalize_sC_barrier"]
        moe_ep = summary["moe_ep_split_rank_major"]
        cake = summary["sm103a_po_and_tcgen05_wait"]

        self.assertEqual(rubin["legacy_corrupted_tiles"], 7)
        self.assertEqual(rubin["fixed_corrupted_tiles"], 0)
        self.assertTrue(moe_ep["all_bridge_cases_passed"])
        self.assertEqual(moe_ep["total_legacy_gemm_evals"], 12288)
        self.assertEqual(moe_ep["total_fixed_gemm_evals"], 6821)
        self.assertEqual(moe_ep["wasted_gemm_reduction_pct"], 44.49)
        self.assertEqual(cake["form_codes"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(cake["peer_inner_transitions"], 224)
        self.assertEqual(cake["peer_outer_transitions"], 4)
        self.assertEqual(cake["peer_transition_reduction_pct"], 98.21)
        self.assertEqual(cake["legacy_tmem_dealloc_hazard_warps"], 9)
        self.assertEqual(cake["fixed_tmem_dealloc_hazard_warps"], 0)

        proc = subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "flashinfer_rubin_moe_barrier_ep_split_ag_matmul"
                    / "run.py"
                ),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS] flashinfer_rubin_moe_barrier_ep_split_ag_matmul", proc.stdout)

    def test_torchtitan_dsv3_mla_rope_megatron_recompute(self) -> None:
        summary = run_torchtitan_megatron()
        mla = summary["dsv3_mla_int32_rope"]
        flex = summary["flex_attention_recompile_guard"]
        ckpt = summary["megatron_recompute_region"]

        self.assertTrue(mla["forward_bitwise_match"])
        self.assertTrue(mla["backward_bitwise_match"])
        self.assertEqual(mla["mem_op_reduction_pct"], 50.0)
        self.assertEqual(mla["fixed_masked_cat_loads_total"], 0)
        self.assertEqual(flex["legacy_eager_fallbacks"], 2)
        self.assertEqual(flex["fixed_eager_fallbacks"], 0)
        self.assertEqual(flex["speedup_vs_legacy"], 3.88)
        self.assertEqual(ckpt["fixed_leaked_te_fp8_queue_entries"], 0)
        self.assertEqual(ckpt["fixed_reference_cycles_per_layer"], 0)
        self.assertEqual(
            ckpt["state_transitions"], ["RECORDED", "RELEASED", "REBUILT", "DONE"]
        )
        self.assertEqual(ckpt["grad_enabled_per_call"], [False, True])

        proc = subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "torchtitan_dsv3_mla_rope_megatron_recompute"
                    / "run.py"
                ),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS] torchtitan_dsv3_mla_rope_megatron_recompute", proc.stdout)

    def test_lmcache_dax_cxl_offset_grpc_mooncake_oplog(self) -> None:
        summary = run_lmcache_mooncake()
        dax = summary["devdax_cxl_l1_offset"]
        grpc = summary["grpc_response_writer_compile"]
        oplog = summary["mooncake_oplog_segment_metrics"]

        self.assertEqual(dax["legacy_sigbus_crashes"], 2)
        self.assertEqual(dax["fixed_sigbus_crashes"], 0)
        self.assertTrue(dax["disjoint_512gib_pair_accepted"])
        self.assertEqual(dax["total_mapped_gib"], 1024)
        self.assertEqual(dax["invalid_cases_rejected"], 5)
        self.assertEqual(grpc["fixed_compiles_structured"], 1)
        self.assertEqual(grpc["already_proto_compiles"], 0)
        self.assertEqual(grpc["compile_reduction_pct"], 99.99)
        self.assertEqual(grpc["speedup_factor"], 24.28)
        self.assertEqual(oplog["legacy_double_counted_entries"], 150)
        self.assertEqual(oplog["fixed_double_counted_entries"], 0)

        proc = subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "lmcache_dax_cxl_offset_grpc_mooncake_oplog"
                    / "run.py"
                ),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS] lmcache_dax_cxl_offset_grpc_mooncake_oplog", proc.stdout)


if __name__ == "__main__":
    unittest.main()
