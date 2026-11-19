from __future__ import annotations

import unittest
from experiments.enterprise_vector_index_zero_downtime_embedding_migration_procrustes_selection.run import build_payload


class TestEnterpriseVectorIndexZeroDowntimeEmbeddingMigrationProcrustesSelection(unittest.TestCase):
    def test_tier4_zero_downtime_low_recall_drop_and_zero_cdc_miss(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertEqual(tier4["production_rag_service_downtime_minutes"], 0.0)
        self.assertLessEqual(tier4["transition_window_recall_at_10_drop_pct"], 0.5)
        self.assertEqual(tier4["stale_cdc_delta_miss_rate_pct"], 0.0)
        self.assertLessEqual(tier4["peak_dual_ram_hnsw_memory_overhead_pct"], 20.0)
        self.assertLess(
            tier4["total_migration_compute_and_memory_tco_usd"],
            tier1["total_migration_compute_and_memory_tco_usd"] * 0.25,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
