import unittest
from experiments.llm_speculative_decoding_kv_page_rollback_shadow_slot_lazy_hash_selection.run import evaluate_spec_decode_kv_rollback_lazy_hash


class TestSpecDecodeKvRollbackLazyHash(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_spec_decode_kv_rollback_lazy_hash()
        self.assertEqual(res["round"], 319)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_preallocated_shadow_slots_device_atomic_seqlen_rewind_plus_lazy_block_hash"]["cpu_kv_manager_overhead_ms_per_step"],
            0.5,
        )
        self.assertGreater(
            tiers["tier4_preallocated_shadow_slots_device_atomic_seqlen_rewind_plus_lazy_block_hash"]["net_speculative_speedup_x"],
            2.3,
        )


if __name__ == "__main__":
    unittest.main()
