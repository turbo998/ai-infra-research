import unittest
from experiments.llm_sft_packing_cross_document_attention_leakage_cu_seqlens_selection.run import (
    evaluate_sequence_packing_cu_seqlens_tiers,
)


class TestSequencePackingCuSeqlensSelection(unittest.TestCase):
    def test_sequence_packing_tiers(self) -> None:
        data = evaluate_sequence_packing_cu_seqlens_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_concatenation_full_causal_mask_and_unreset_rope"]
        t4 = tiers["tier4_varlen_cu_seqlens_plus_rope_reset_and_boundary_ignore_index"]
        self.assertLess(t1["downstream_sft_accuracy_pct"], 77.0)
        self.assertEqual(t4["downstream_sft_accuracy_pct"], 82.0)
        self.assertEqual(t4["extra_2d_mask_hbm_gb"], 0.0)
        self.assertLess(t4["attention_kernel_latency_ms"], t1["attention_kernel_latency_ms"])


if __name__ == "__main__":
    unittest.main()
