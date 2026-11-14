import unittest
from experiments.ai_agent_memory_memgpt_bitemporal_graph_selection.run import (
    evaluate_agent_memory_tiers,
)


class TestAgentMemorySelection(unittest.TestCase):
    def test_tiers_and_temporal_accuracy(self) -> None:
        data = evaluate_agent_memory_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertGreater(data["workload"]["cost_reduction_pct"], 90.0)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unbounded_full_history_context_stuffing"]
        t3 = tiers["tier3_flat_vector_rag_conversation_chunks"]
        t4 = tiers["tier4_os_paged_core_memory_plus_bitemporal_graph"]
        self.assertGreater(t1["cost_per_100_turns_usd"], 4.0)
        self.assertLess(t3["temporal_state_update_accuracy_pct"], 60.0)
        self.assertGreaterEqual(t4["temporal_state_update_accuracy_pct"], 95.0)


if __name__ == "__main__":
    unittest.main()
