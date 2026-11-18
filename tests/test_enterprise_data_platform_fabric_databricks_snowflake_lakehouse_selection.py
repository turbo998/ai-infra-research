import unittest
from experiments.enterprise_data_platform_fabric_databricks_snowflake_lakehouse_selection.run import build_payload


class TestEnterpriseDataPlatformFabricDatabricksSnowflakeLakehouseSelection(unittest.TestCase):
    def test_tier4_achieves_zero_copy_subsecond_bi_and_lowest_tco(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_siloed_adf_adls_dw_import_duplication"]
        t3 = tiers["tier3_pure_spark_lakehouse_plus_directquery_bi_bottleneck"]
        t4 = tiers["tier4_open_onelake_uniform_delta_iceberg_plus_databricks_etl_plus_fabric_direct_lake"]

        self.assertEqual(t4["storage_and_etl_copy_duplication_factor"], 1.0)
        self.assertLessEqual(t4["end_to_end_data_freshness_lag_minutes"], 3.0)
        self.assertLessEqual(t4["bi_and_ai_copilot_p95_query_latency_sec"], 0.5)
        self.assertGreater(t3["bi_and_ai_copilot_p95_query_latency_sec"], 5.0)
        self.assertLess(t4["monthly_platform_compute_and_storage_tco_usd"], t1["monthly_platform_compute_and_storage_tco_usd"] * 0.35)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
