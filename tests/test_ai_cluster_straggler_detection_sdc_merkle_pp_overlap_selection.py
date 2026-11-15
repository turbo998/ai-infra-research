import unittest
from experiments.ai_cluster_straggler_detection_sdc_merkle_pp_overlap_selection.run import (
    evaluate_sdc_and_straggler_detection_tiers,
)


class TestSdcAndStragglerDetectionSelection(unittest.TestCase):
    def test_sdc_straggler_tiers(self) -> None:
        data = evaluate_sdc_and_straggler_detection_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_profiler_sort_by_longest_nccl_kernel_duration"]
        t4 = tiers["tier4_online_cuda_event_compute_wait_split_plus_pre_allreduce_grad_and_replica_weight_hash"]
        self.assertEqual(t1["straggler_and_sdc_pinpoint_accuracy_pct"], 0.0)
        self.assertEqual(t4["straggler_and_sdc_pinpoint_accuracy_pct"], 100.0)
        self.assertLess(t4["mean_time_to_isolate_faulty_gpu_sec"], 2.0)


if __name__ == "__main__":
    unittest.main()
