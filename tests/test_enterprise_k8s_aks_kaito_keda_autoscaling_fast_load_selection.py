"""Unit tests for Round 373 Enterprise Kubernetes (AKS/EKS) LLM Autoscaling, KAITO/KEDA & Model Asset Fast Cold-Start Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_k8s_aks_kaito_keda_autoscaling_fast_load_selection.run import (
    evaluate_enterprise_k8s_aks_kaito_keda_autoscaling_fast_load,
)


class TestEnterpriseK8sAksKaitoKedaAutoscalingFastLoadSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_k8s_aks_kaito_keda_autoscaling_fast_load()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_k8s_aks_kaito_keda_autoscaling_fast_load_selection",
        )
        self.assertEqual(data["round"], 373)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLessEqual(
            tiers["tier3_kaito_keda_p2p_image_local_nvme_runai_streamer"]["scale_out_ready_seconds"],
            40.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_coldstart_speedup_x"],
            100.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_k8s_aks_kaito_keda_autoscaling_fast_load_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_k8s_aks_kaito_keda_autoscaling_fast_load()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
