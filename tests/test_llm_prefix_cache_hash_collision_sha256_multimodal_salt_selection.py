import unittest
from experiments.llm_prefix_cache_hash_collision_sha256_multimodal_salt_selection.run import (
    evaluate_prefix_cache_block_hash_tiers,
)


class TestPrefixCacheBlockHashSelection(unittest.TestCase):
    def test_prefix_cache_hash_tiers(self) -> None:
        data = evaluate_prefix_cache_block_hash_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_builtin_python_hash_random_seed_zero_distributed_hit"]
        t2 = tiers["tier2_token_id_only_hash_multimodal_image_placeholder_poisoning"]
        t4 = tiers["tier4_merkle_sha256_blake3_with_multimodal_lora_and_tenant_salt"]
        self.assertEqual(t1["distributed_kv_hit_rate_pct"], 0.0)
        self.assertGreater(t2["multimodal_or_lora_cross_poison_rate_pct"], 20.0)
        self.assertEqual(t4["multimodal_or_lora_cross_poison_rate_pct"], 0.0)
        self.assertTrue(t4["cryptographic_tenant_isolation"])


if __name__ == "__main__":
    unittest.main()
