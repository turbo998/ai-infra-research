import unittest
from experiments.moe_expert_offloading_cpu_pinned_pcie_prefetch_selection.run import (
    evaluate_moe_expert_offloading_prefetch_tiers,
)


class TestMoeExpertOffloadingPrefetchSelection(unittest.TestCase):
    def test_moe_expert_offload_tiers(self) -> None:
        data = evaluate_moe_expert_offloading_prefetch_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_reactive_on_demand_pcie_expert_loading_stall"]
        t4 = tiers["tier4_hot_vram_cache_plus_speculative_prefetch_and_tail_skip"]
        self.assertGreater(t1["decode_tpot_ms"], 300.0)
        self.assertLess(t4["decode_tpot_ms"], 20.0)
        self.assertGreater(t4["effective_vram_hit_or_hidden_pct"], 95.0)


if __name__ == "__main__":
    unittest.main()
