from __future__ import annotations

import unittest
from experiments.enterprise_pipeline_midstream_leak_ili_corrosion_drilling_agent_selection.run import build_payload


class TestEnterprisePipelineMidstreamLeakIliCorrosionDrillingAgentSelection(unittest.TestCase):
    def test_tier4_rttm_leak_speed_b31g_dig_precision_and_stuck_pipe(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["pinhole_leak_detection_time_minutes"], 2.0)
        self.assertLessEqual(tier4["false_emergency_shutin_rate_pct"], 0.5)
        self.assertLessEqual(tier4["unnecessary_integrity_dig_overkill_pct"], 1.5)
        self.assertLessEqual(tier4["drilling_stuck_pipe_npt_hours_per_month"], 4.0)
        self.assertLess(tier4["monthly_shutin_dig_and_npt_loss_usd"], tier1["monthly_shutin_dig_and_npt_loss_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
