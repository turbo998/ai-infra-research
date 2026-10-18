from __future__ import annotations

import unittest

from experiments.multi_lora_serving.run import (
    evaluate_batching_scalability,
    evaluate_rank_crossover,
    evaluate_unified_paging,
    evaluate_zipfian_prefetch,
    run_simulation,
    validate_results,
)


class TestMultiLoraServing(unittest.TestCase):
    def test_punica_sgmv_scalability(self) -> None:
        rows = evaluate_batching_scalability()
        by_k = {r.active_adapters: r for r in rows}
        self.assertGreater(by_k[64].throughput_speedup, 30.0)
        self.assertGreater(by_k[128].throughput_speedup, 60.0)

    def test_slora_unified_paging_and_prefetch(self) -> None:
        paging = evaluate_unified_paging()
        self.assertLess(paging[2].external_fragmentation_pct, 2.5)
        self.assertGreater(paging[2].max_concurrent_kv_tokens, paging[0].max_concurrent_kv_tokens)

        prefetch = evaluate_zipfian_prefetch()
        self.assertGreater(prefetch[2].cache_hit_rate_pct, 94.0)
        self.assertLess(prefetch[2].p99_ttft_ms, prefetch[1].p99_ttft_ms)

    def test_rank_crossover_r_star(self) -> None:
        crossover = {r.lora_rank: r for r in evaluate_rank_crossover()}
        self.assertGreaterEqual(
            crossover[64].punica_sgmv_tok_s, crossover[64].merged_weight_multi_instance_tok_s
        )
        self.assertLess(
            crossover[96].punica_sgmv_tok_s, crossover[96].merged_weight_multi_instance_tok_s
        )

    def test_full_simulation_validation(self) -> None:
        data = run_simulation()
        validate_results(data)


if __name__ == "__main__":
    unittest.main()
