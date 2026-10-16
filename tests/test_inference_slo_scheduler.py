"""Unit tests for the vLLM V1 unified scheduler, KV block allocator, and P/D SLO experiment."""

import json
import unittest

from experiments.inference_slo_scheduler.run import (
    HERE,
    calculate,
    evaluate_inference_slo_case,
    main,
)


class InferenceSloSchedulerTest(unittest.TestCase):
    def test_fixture_check_passes(self) -> None:
        self.assertEqual(main(["--check"]), 0)

    def test_vllm_v1_scheduler_and_kv_invariants(self) -> None:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        output = calculate(fixture)
        by_id = {item["result"]["case_id"]: item["result"] for item in output["cases"]}

        large_chunk = by_id["colocated_large_chunk_tpot_violation"]
        self.assertFalse(large_chunk["colocated_meets_tpot_slo"])
        self.assertTrue(large_chunk["colocated_meets_ttft_slo"])
        self.assertTrue(large_chunk["pd_meets_tpot_slo"])
        self.assertTrue(large_chunk["pd_meets_ttft_slo"])

        small_chunk = by_id["colocated_small_chunk_tpot_safe_ttft_inflated"]
        self.assertTrue(small_chunk["colocated_meets_tpot_slo"])
        self.assertFalse(small_chunk["colocated_meets_ttft_slo"])
        self.assertEqual(small_chunk["preallocated_extra_blocks"], 4)

        full_hit = by_id["full_prefix_cache_hit_last_block_recompute"]
        self.assertTrue(full_hit["full_prefix_hit_recompute_triggered"])
        self.assertEqual(full_hit["tokens_to_prefill_total"], 16)
        self.assertEqual(full_hit["effective_computed_tokens"], 4080)

        evictable_guard = by_id["evictable_prefix_hit_blocks_new_allocation"]
        self.assertEqual(evictable_guard["free_blocks_initial"], 40)
        self.assertEqual(evictable_guard["usable_free_blocks_after_evictable_guard"], 20)
        self.assertEqual(evictable_guard["num_new_blocks_needed_step1"], 28)
        self.assertFalse(evictable_guard["can_allocate_waiting_step1"])

        preempt = by_id["zero_free_blocks_running_preemption"]
        self.assertTrue(preempt["running_Needs_block_triggers_preemption"])
        self.assertEqual(preempt["free_blocks_after_one_preemption"], 128)

    def test_rejects_invalid_cached_tokens(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_inference_slo_case(
                case_id="bad_cache",
                block_size=16,
                num_gpu_blocks=100,
                max_model_len=1024,
                max_num_batched_tokens=256,
                running_decode_requests=4,
                running_occupied_blocks=20,
                lowest_priority_running_req_blocks=4,
                waiting_prompt_tokens=64,
                waiting_cached_prefix_blocks=8,
                evictable_cached_prefix_blocks=0,
                kv_bytes_per_token=1024,
                base_decode_step_ms=10.0,
                marginal_prefill_token_ms=0.01,
                pd_rdma_bandwidth_bytes_per_sec=1000000000,
                tpot_slo_ms=25.0,
                ttft_slo_ms=100.0,
            )


if __name__ == "__main__":
    unittest.main()
