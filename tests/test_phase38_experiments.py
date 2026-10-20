import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashinfer_cake_ssd_vcount_mla_tmem_fmha_planfree import run as exp1
from experiments.sglang_dspark_sim_accept_radix_reclaimable_subpool_vllm_rocm import (
    run as exp2,
)
from experiments.torchtitan_dsv3_hf_save_map_gpt_oss_mxfp4_dynamo_elastic_ep import (
    run as exp3,
)


class TestPhase38Experiments(unittest.TestCase):
    def test_exp1_flashinfer_cake_ssd_mla_fmha(self) -> None:
        summary = exp1.build_summary()
        exp1.verify_summary(summary)
        self.assertEqual(exp1.chunk_parallel_vcount((10, 7), 208), 208)
        self.assertEqual(exp1.chunk_parallel_vcount((10, 7), 216), 212)
        self.assertEqual(exp1.chunk_parallel_vcount((10, 3), 148), 148)
        self.assertEqual(exp1.chunk_parallel_vcount((10, 0), 1024), 148)
        self.assertEqual(
            exp1.dcp_balanced_program(
                "dcp_spec_bf16_fp8_balanced",
                batch_size=1,
                num_kv_heads=8,
                max_pages_per_seq=36,
                sm_count=148,
            ),
            2,
        )
        self.assertEqual(
            exp1.dcp_balanced_program(
                "dcp_spec_bf16_fp8_balanced",
                batch_size=1,
                num_kv_heads=8,
                max_pages_per_seq=132,
                sm_count=148,
            ),
            1,
        )

    def test_exp2_sglang_dspark_radix_subpool_vllm_rocm(self) -> None:
        summary = exp2.build_summary()
        exp2.verify_summary(summary)
        self.assertEqual(
            summary["dspark_simulated_acceptance"][
                "cuda_simulated_softmax_elements_saved_per_step"
            ],
            24330240,
        )
        self.assertTrue(
            summary["radix_reclaimable_and_subpool"][
                "negative_adjust_rejected_without_mutation"
            ]
        )
        self.assertTrue(
            summary["vllm_rocm_gated_rmsnorm"]["rocm_unquantized_default_enabled"]
        )

    def test_exp3_torchtitan_dynamo_mooncake_nixl(self) -> None:
        summary = exp3.build_summary()
        exp3.verify_summary(summary)
        dsv3 = summary["torchtitan_adapters"]["deepseek_v3_index_filtering"]
        self.assertEqual(dsv3[0]["filtered_index_entries"], 45395)
        self.assertEqual(dsv3[1]["filtered_index_entries"], 46183)
        self.assertTrue(
            summary["dynamo_elastic_ep_and_snapshot"]["elastic_ep_scenarios"][1][
                "first_moe_forward_succeeds"
            ]
        )
        self.assertTrue(
            summary["dynamo_elastic_ep_and_snapshot"]["trtllm_snapshot_transports"][2][
                "post_restore_first_collective_ok"
            ]
        )

    def test_cli_check_flags(self) -> None:
        root = Path(__file__).resolve().parent.parent
        for rel in (
            "experiments/flashinfer_cake_ssd_vcount_mla_tmem_fmha_planfree/run.py",
            "experiments/sglang_dspark_sim_accept_radix_reclaimable_subpool_vllm_rocm/run.py",
            "experiments/torchtitan_dsv3_hf_save_map_gpt_oss_mxfp4_dynamo_elastic_ep/run.py",
        ):
            proc = subprocess.run(
                [sys.executable, str(root / rel), "--check"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn('"status": "ok"', proc.stdout)


if __name__ == "__main__":
    unittest.main()
