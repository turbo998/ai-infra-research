import unittest
from experiments.enterprise_industrial_predictive_maintenance_scada_copilot_selection.run import build_payload


class TestEnterpriseIndustrialPredictiveMaintenanceScadaCopilotSelection(unittest.TestCase):
    def test_tier4_maximizes_diagnosis_accuracy_and_minimizes_mttr(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_static_scada_threshold_paper_manuals"]
        t2 = tiers["tier2_raw_sensor_csv_stuffed_into_text_llm"]
        t4 = tiers["tier4_edge_fft_timesfm_plus_colpali_schematic_rag_plus_cmms_agent"]

        self.assertGreater(t4["root_cause_diagnosis_accuracy_pct"], 92.0)
        self.assertLess(t2["root_cause_diagnosis_accuracy_pct"], t1["root_cause_diagnosis_accuracy_pct"])
        self.assertLessEqual(t4["false_alarm_fatigue_rate_pct"], 3.0)
        self.assertLessEqual(t4["mean_time_to_repair_hours"], 1.5)
        self.assertLess(t4["monthly_unplanned_downtime_and_ops_cost_usd"], t1["monthly_unplanned_downtime_and_ops_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
