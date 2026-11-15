import unittest
from experiments.llm_structured_output_xgrammar_outlines_fsm_mask_cache_selection.run import (
    evaluate_structured_output_xgrammar_tiers,
)


class TestStructuredOutputXgrammarSelection(unittest.TestCase):
    def test_structured_output_tiers(self) -> None:
        data = evaluate_structured_output_xgrammar_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t2 = tiers["tier2_eager_full_vocab_fsm_compilation_outlines_legacy"]
        t4 = tiers["tier4_xgrammar_bitmask_plus_jump_forward_deterministic_fast_forward"]
        self.assertGreater(t2["dynamic_schema_compile_ttft_ms"], 3000.0)
        self.assertLess(t4["dynamic_schema_compile_ttft_ms"], 15.0)
        self.assertGreater(t4["end_to_end_json_speedup_vs_unconstrained_x"], 1.8)


if __name__ == "__main__":
    unittest.main()
