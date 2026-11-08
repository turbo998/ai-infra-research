import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.rlhf_post_training_infra_selection.run import RESULTS_PATH, compute_results


class TestRlhfPostTrainingInfraSelection(unittest.TestCase):
    def test_deterministic_model_and_check_flag(self) -> None:
        script = Path(__file__).resolve().parent.parent / "experiments" / "rlhf_post_training_infra_selection" / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data, compute_results())
        self.assertEqual(data["metadata"]["provenance_type"], "synthetic")
        self.assertFalse(data["metadata"]["independently_reproduced"])
        self.assertEqual(data["metadata"]["physical_gpu_cluster_measurement"], "not_measured")

    def test_rlhf_grpo_tiers_and_straggler_bubble_invariants(self) -> None:
        data = compute_results()
        t1 = data["tiers"]["tier1_naive_fsdp_colocated_rollout"]
        t2 = data["tiers"]["tier2_static_disaggregated_ray_pools"]
        t3 = data["tiers"]["tier3_hybrid_engine_colocated_resharding"]
        t4 = data["tiers"]["tier4_async_partial_rollout_streaming"]

        # Step time monotonic reduction: 612.0s -> 286.5s -> 158.2s -> 96.4s
        self.assertEqual(t1["total_step_time_sec"], 612.0)
        self.assertEqual(t2["total_step_time_sec"], 286.5)
        self.assertEqual(t3["total_step_time_sec"], 158.2)
        self.assertEqual(t4["total_step_time_sec"], 96.4)

        # Speedup invariants
        self.assertEqual(t2["speedup_vs_tier1"], 2.14)
        self.assertEqual(t3["speedup_vs_tier1"], 3.87)
        self.assertEqual(t4["speedup_vs_tier1"], 6.35)
        self.assertEqual(t4["speedup_vs_tier3"], 1.64)

        # Partial Rollout + Async Streaming slashes straggler bubble from 52.0% -> 8.5%
        self.assertEqual(t3["rollout_straggler_idle_bubble_pct"], 52.0)
        self.assertEqual(t4["rollout_straggler_idle_bubble_pct"], 8.5)

        # 1,000-step TCO savings invariants
        self.assertEqual(t1["cost_per_1000_steps_usd"], 27200.00)
        self.assertEqual(t3["cost_per_1000_steps_usd"], 7030.40)
        self.assertEqual(t4["cost_per_1000_steps_usd"], 4284.80)
        self.assertEqual(t4["savings_per_1000_steps_vs_tier1_usd"], 22915.20)
        self.assertEqual(t4["savings_per_1000_steps_vs_tier3_usd"], 2745.60)


if __name__ == "__main__":
    unittest.main()
