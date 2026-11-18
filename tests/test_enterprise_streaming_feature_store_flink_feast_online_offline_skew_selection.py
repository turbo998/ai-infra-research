from __future__ import annotations

import unittest
from experiments.enterprise_streaming_feature_store_flink_feast_online_offline_skew_selection.run import build_payload


class TestEnterpriseStreamingFeatureStoreFlinkFeastOnlineOfflineSkewSelection(unittest.TestCase):
    def test_tier4_zero_skew_zero_leakage_and_sub_5ms_p99(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertEqual(tier4["online_offline_feature_skew_pct"], 0.0)
        self.assertEqual(tier4["future_data_leakage_rate_pct"], 0.0)
        self.assertLessEqual(tier4["online_feature_serving_p99_ms"], 5.0)
        self.assertLessEqual(tier4["feature_freshness_lag_sec"], 0.5)
        self.assertLess(tier4["monthly_infra_and_fraud_slippage_cost_usd"], tier1["monthly_infra_and_fraud_slippage_cost_usd"] * 0.2)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
