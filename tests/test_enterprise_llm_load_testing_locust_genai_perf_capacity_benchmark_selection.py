from __future__ import annotations

import unittest
from experiments.enterprise_llm_load_testing_locust_genai_perf_capacity_benchmark_selection.run import build_payload


class TestEnterpriseLlmLoadTestingLocustGenaiPerfCapacityBenchmarkSelection(unittest.TestCase):
    def test_tier4_minimizes_sizing_error_and_maximizes_slo_goodput(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLess(tier4["benchmark_to_production_capacity_error_pct"], 5.0)
        self.assertLess(
            tier4["benchmark_to_production_capacity_error_pct"],
            tier2["benchmark_to_production_capacity_error_pct"] * 0.10,
        )
        self.assertEqual(tier4["post_launch_sla_breach_incident_rate_pct"], 0.0)
        self.assertGreater(tier4["goodput_valid_requests_meeting_ttft_tpot_sla_pct"], 99.0)
        self.assertGreater(
            tier4["goodput_valid_requests_meeting_ttft_tpot_sla_pct"],
            tier3["goodput_valid_requests_meeting_ttft_tpot_sla_pct"] + 35.0,
        )
        self.assertLess(
            tier4["monthly_overprovisioning_or_outage_penalty_usd"],
            tier1["monthly_overprovisioning_or_outage_penalty_usd"] * 0.10,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
