import unittest
from experiments.cuda_green_contexts_sm_partition_stream_overlap_selection.run import (
    evaluate_green_context_sm_tiers,
)


class TestCudaGreenContextsSmPartitionSelection(unittest.TestCase):
    def test_green_context_tiers(self) -> None:
        data = evaluate_green_context_sm_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_default_equal_priority_multi_stream_contention"]
        t4 = tiers["tier4_cuda12_green_contexts_intra_process_sm_split"]
        self.assertGreater(t1["comm_stream_launch_delay_ms"], 2.0)
        self.assertLessEqual(t4["comm_stream_launch_delay_ms"], 0.05)
        self.assertEqual(t4["compute_sms"] + t4["comm_decode_sms"], 132)


if __name__ == "__main__":
    unittest.main()
