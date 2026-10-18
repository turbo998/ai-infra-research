from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.disaggregated_kv_routing.run import (
    DEFAULT_INPUT,
    DEFAULT_JSON,
    deadline_mlu,
    evaluate_fixture,
    main,
    order_by_urgency,
    plan_rdma_slices,
)


class DisaggregatedKvRoutingTest(unittest.TestCase):
    def test_committed_outputs_match_check_flag(self) -> None:
        self.assertEqual(main(["--check"]), 0)

    def test_mooncake_plan_rdma_slices_tail_folding(self) -> None:
        # 1,142,341,632 bytes with 4 MiB base_block and max_slices=16 -> block_size = 75,497,472 (72 MiB)
        # 15 * 72 MiB = 1,132,462,080; tail = 9,879,552 bytes (13.08% < 25% merge_ratio) -> folded into slice 15!
        res = plan_rdma_slices(1142341632, base_block=4194304, max_slices=16, merge_ratio=0.25)
        self.assertEqual(res["block_size"], 75497472)
        self.assertEqual(res["count"], 15)
        self.assertEqual(res["last_slice_bytes"], 85377024)

    def test_mooncake_order_by_urgency_eliminates_head_of_line_miss(self) -> None:
        fixture = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        results = evaluate_fixture(fixture)
        arb4 = results["deadline_arbitration"][0]
        self.assertEqual(arb4["fifo_slo_misses"], 1)
        self.assertEqual(arb4["equal_split_slo_misses"], 1)
        self.assertEqual(arb4["mooncake_urgent_slo_misses"], 0)
        self.assertEqual(arb4["ordered_flow_ids"][0], "flow3_voice_agent_16mb_urgent_1_5ms")

        arb6 = results["deadline_arbitration"][1]
        self.assertEqual(arb6["fifo_slo_misses"], 3)
        self.assertEqual(arb6["equal_split_slo_misses"], 5)
        self.assertEqual(arb6["mooncake_urgent_slo_misses"], 0)

    def test_sglang_hicache_pd_pipeline_regimes(self) -> None:
        results = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
        cases = {c["case_id"]: c for c in results["hicache_pd_pipeline"]}

        l2_case = cases["agentic_64k_l2_host_hit_overlap"]
        self.assertEqual(l2_case["decode_prefix_len"], 49152)
        self.assertEqual(l2_case["restore_token_count"], 40960)
        self.assertGreater(l2_case["ttft_speedup_x"], 2.5)

        # Counterexample: when 24,576 tokens must be fetched from a 100 Gbps L3 storage tier before L2->L1 load_back,
        # the local restore branch (951.03 ms) becomes slower than recomputing + RDMA-transferring from Prefill (694.38 ms).
        l3_case = cases["multiturn_64k_l2_plus_l3_nvme_hit"]
        self.assertLess(l3_case["ttft_speedup_x"], 1.0)
        self.assertEqual(l3_case["bottleneck"], "local_hicache_restore")

    def test_invalid_inputs_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            plan_rdma_slices(-1, 4096, 8)
        self.assertEqual(deadline_mlu(0, 1024, 0, 100, 1e9), 0.0)
        self.assertTrue(deadline_mlu(0, 1024, 100, 100, 1e9) == float("inf"))
        order = order_by_urgency(
            [
                {"length": 1000, "deadline_ns": 5000},
                {"length": 1000, "deadline_ns": 1000},
            ],
            now_ns=0,
            bw_bps=1e9,
        )
        self.assertEqual(order, [1, 0])


if __name__ == "__main__":
    unittest.main()
