import unittest
from experiments.llm_prefill_causal_ring_attention_striped_zigzag_load_balance_selection.run import (
    evaluate_causal_ring_attention_zigzag_tiers,
)


class TestCausalRingAttentionZigZagSelection(unittest.TestCase):
    def test_zigzag_ring_attention_tiers(self) -> None:
        data = evaluate_causal_ring_attention_zigzag_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_contiguous_chunk_ring_attention_causal_bubble"]
        t4 = tiers["tier4_per_document_varlen_zigzag_ring_attention_packed_sequences"]
        self.assertGreater(t1["gpu_idle_bubble_pct"], 40.0)
        self.assertEqual(t4["max_to_avg_flop_imbalance_ratio"], 1.00)
        self.assertLess(t4["prefill_128k_latency_ms"], 1000.0)


if __name__ == "__main__":
    unittest.main()
