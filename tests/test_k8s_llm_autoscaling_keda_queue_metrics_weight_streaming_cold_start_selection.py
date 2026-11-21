import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestK8sLlmAutoscalingKedaWeightStreamingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "k8s_llm_autoscaling_keda_queue_metrics_weight_streaming_cold_start_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_script_check_and_invariants(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"], "synthetic + anonymized_reported")
        self.assertFalse(data["independently_reproduced"])
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_k8s_hpa_on_gpu_util_plus_monolithic_oci_image_pull"]
        t4 = tiers["tier4_keda_predictive_kv_trigger_plus_nvme_gds_and_vllm_sleep_level1_warm_pool"]
        self.assertLess(t4["replica_scale_out_ready_sec"], 15.0)
        self.assertLess(
            t4["monthly_cluster_elastic_tco_usd"],
            t1["monthly_cluster_elastic_tco_usd"] * 0.40,
        )

    def test_artifacts_exist(self) -> None:
        slug = "k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
