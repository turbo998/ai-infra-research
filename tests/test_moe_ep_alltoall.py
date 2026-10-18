from __future__ import annotations

import json
import unittest

from experiments.moe_ep_alltoall.run import (
    DEFAULT_INPUT,
    DEFAULT_JSON,
    compute_token_layout_bytes,
    evaluate_fixture,
    main,
)


class MoeEpAllToAllTest(unittest.TestCase):
    def test_committed_outputs_match_check_flag(self) -> None:
        self.assertEqual(main(["--check"]), 0)

    def test_deepep_token_layout_exact_alignment(self) -> None:
        # DeepSeek-V3 Dispatch: hidden=7168 (FP8 -> 7168 B), sf=56*4=224 B, topk=8, with_metadata=True
        # metadata = 8*(4+4) + 9*4 = 100 B -> aligned(32) = 128 B
        # data = align(7168 + 224 + 128, 64) = align(7520, 64) = 7552 B
        # with mbarrier (8 B -> aligned(32) = 32 B) = 7584 B
        disp = compute_token_layout_bytes(7168, 224, 8, with_metadata=True, with_mbarrier=True)
        self.assertEqual(disp["num_metadata_bytes"], 100)
        self.assertEqual(disp["num_data_bytes_rdma_aligned"], 7552)
        self.assertEqual(disp["smem_slot_bytes_with_mbarrier"], 7584)

        # DeepSeek-V3 Combine: hidden=7168 (BF16 -> 14336 B), sf=0, topk=8, with_metadata=False
        # metadata = 8*8 = 64 B -> aligned(32) = 64 B
        # data = align(14336 + 0 + 64, 64) = 14400 B; with mbarrier = 14432 B
        comb = compute_token_layout_bytes(14336, 0, 8, with_metadata=False, with_mbarrier=True)
        self.assertEqual(comb["num_data_bytes_rdma_aligned"], 14400)
        self.assertEqual(comb["smem_slot_bytes_with_mbarrier"], 14432)

    def test_hierarchical_deduplication_saves_rdma_bandwidth(self) -> None:
        fixture = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        results = evaluate_fixture(fixture)
        cases = {c["case_id"]: c for c in results["hierarchical_deduplication"]}

        group4 = cases["dsv3_ep64_8nodes_x_8gpus_group_limited_4nodes"]
        self.assertEqual(group4["rdma_traffic_reduction_x"], 2.2)
        self.assertGreater(group4["end_to_end_speedup_x"], 2.0)

        gb200 = cases["gb200_ep64_2racks_x_32gpus_hybrid"]
        self.assertGreater(gb200["rdma_traffic_reduction_x"], 4.0)

    def test_eplb_skew_mitigation_and_over_replication_counterexample(self) -> None:
        results = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
        cases = {c["case_id"]: c for c in results["eplb_and_skew"]}

        skew1 = cases["ep32_256experts_zipf_0_85_with_1_redundant_per_rank"]
        self.assertGreater(skew1["step_speedup_x"], 2.5)
        self.assertLess(skew1["imbalance_ratio_with_eplb"], skew1["imbalance_ratio_no_eplb"])

        # Counterexample: over-replicating 4 redundant experts per rank during a small 4096-token batch
        # incurs 5.835 ms of NVLink weight prefetching vs 0.420 ms of unbalanced compute (0.07x slowdown).
        counter = cases["ep32_256experts_microbatch_over_replicated_counterexample"]
        self.assertLess(counter["step_speedup_x"], 0.1)

    def test_invalid_token_layout_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            compute_token_layout_bytes(7100, 0, 8, True, False)  # not divisible by 32
        with self.assertRaises(ValueError):
            compute_token_layout_bytes(7168, 0, 64, True, False)  # topk > 32


if __name__ == "__main__":
    unittest.main()
