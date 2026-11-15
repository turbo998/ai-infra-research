import unittest
from experiments.llm_activation_checkpointing_selective_op_sac_offload_memory_selection.run import evaluate_selective_activation_checkpointing


class TestSelectiveActivationCheckpointing(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_selective_activation_checkpointing()
        self.assertEqual(res["round"], 298)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_op_level_sac_must_save_mm_flash_recompute_swiglu_norm_rope"]["activation_memory_gb"],
            20.0,
        )
        self.assertGreater(
            tiers["tier4_op_level_sac_must_save_mm_flash_recompute_swiglu_norm_rope"]["effective_training_tflops"],
            tiers["tier2_full_layer_activation_checkpointing_33pct_flops_tax"]["effective_training_tflops"] * 1.25,
        )


if __name__ == "__main__":
    unittest.main()
