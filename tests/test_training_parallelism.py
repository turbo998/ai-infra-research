"""Unit tests for the 6D training parallelism, FSDP2 transient buffers, CP causal balance, and DeepEP experiment."""

import json
import unittest

from experiments.training_parallelism.run import (
    HERE,
    calculate,
    evaluate_parallelism_case,
    main,
)


class TrainingParallelismTest(unittest.TestCase):
    def test_fixture_check_passes(self) -> None:
        self.assertEqual(main(["--check"]), 0)

    def test_6d_mesh_and_fsdp2_invariants(self) -> None:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        output = calculate(fixture)
        by_id = {item["result"]["case_id"]: item["result"] for item in output["cases"]}

        fsdp_1d = by_id["dense_70b_1d_fsdp_only_oom"]
        hsdp_2d = by_id["dense_70b_2d_hsdp_tp8_sp"]
        non_dim0 = by_id["dense_70b_2d_tp_fsdp_non_dim0_penalty"]
        cp8 = by_id["long_ctx_64k_cp8_round_robin_vs_contiguous"]
        moe_ep8 = by_id["moe_64e_ep8_compact_vs_expanded"]

        # 1D FSDP vs 2D FSDP+TP+SP have identical static state, but 1D OOMs due to unsharded activations & transient buffers.
        self.assertEqual(
            fsdp_1d["memory_breakdown"]["static_state_gib"],
            hsdp_2d["memory_breakdown"]["static_state_gib"],
        )
        self.assertTrue(fsdp_1d["memory_breakdown"]["exceeds_hbm_capacity"])
        self.assertFalse(hsdp_2d["memory_breakdown"]["exceeds_hbm_capacity"])
        self.assertLess(
            hsdp_2d["communication_Step_metrics"]["total_comm_ms"],
            fsdp_1d["communication_Step_metrics"]["total_comm_ms"],
        )

        # Shard(dim != 0) allocates an extra 1x layer buffer during copy_out.
        self.assertGreater(
            non_dim0["memory_breakdown"]["non_dim0_extra_copy_out_mib"],
            0.0,
        )
        self.assertAlmostEqual(
            non_dim0["memory_breakdown"]["fsdp_transient_peak_gib"]
            / hsdp_2d["memory_breakdown"]["fsdp_transient_peak_gib"],
            4.0 / 3.0,
            places=3,
        )

        # CP=8 participates in FSDP weight sharding (fsdp_mesh_size = dp_shard * cp = 8) and fixes 15:1 causal skew.
        self.assertEqual(cp8["mesh"]["fsdp_mesh_size"], 8)
        self.assertEqual(cp8["cp_causal_balance"]["cp_contiguous_max_to_min_skew"], 15.0)
        self.assertEqual(cp8["cp_causal_balance"]["cp_round_robin_skew"], 1.0)

        # MoE efsdp folding and DeepEP v2 compact deduplication savings.
        self.assertEqual(moe_ep8["mesh"]["efsdp_mesh_size"], 4)
        self.assertEqual(moe_ep8["mesh"]["dense_shard_divisor"], moe_ep8["mesh"]["moe_shard_divisor"])
        self.assertLess(
            moe_ep8["deepep_metrics"]["expected_compact_sends_per_token"],
            8.0,
        )
        self.assertGreater(moe_ep8["deepep_metrics"]["dedup_savings_ratio"], 0.30)

    def test_rejects_invalid_mesh_product(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_parallelism_case(
                case_id="bad_mesh",
                world_size=64,
                pp=2,
                dp_replicate=2,
                dp_shard=4,
                cp=2,
                tp=4,
                num_layers=80,
                hidden_size=8192,
                microbatch_size=1,
                num_microbatches=8,
                seq_len=4096,
                dense_params_per_layer=1024,
                intra_node_bw_bytes_per_sec=400_000_000_000,
                inter_node_bw_bytes_per_sec=50_000_000_000,
            )


if __name__ == "__main__":
    unittest.main()
