from __future__ import annotations

import unittest
from experiments.enterprise_adaptive_learning_knowledge_tracing_socratic_tutor_selection.run import build_payload


class TestEnterpriseAdaptiveLearningKnowledgeTracingSocraticTutorSelection(unittest.TestCase):
    def test_tier4_retention_zero_spoiler_and_sandbox_grader(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier4["ninety_day_skill_retention_pct"], 85.0)
        self.assertEqual(tier4["direct_answer_spoiler_leak_rate_pct"], 0.0)
        self.assertEqual(tier4["lab_grading_false_pass_rate_pct"], 0.0)
        self.assertLessEqual(tier4["time_to_competency_hours"], 12.0)
        self.assertLess(tier4["monthly_training_and_incident_rework_cost_usd"], tier1["monthly_training_and_incident_rework_cost_usd"] * 0.15)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
