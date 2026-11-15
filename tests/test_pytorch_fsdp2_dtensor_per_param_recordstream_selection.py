import unittest
from experiments.pytorch_fsdp2_dtensor_per_param_recordstream_selection.run import (
    evaluate_fsdp2_dtensor_recordstream_tiers,
)


class TestPyTorchFSDP2DTensorRecordStreamSelection(unittest.TestCase):
    def test_fsdp2_dtensor_tiers(self) -> None:
        data = evaluate_fsdp2_dtensor_recordstream_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_fsdp1_flat_parameter_record_stream_allocator_bloat"]
        t4 = tiers["tier4_fsdp2_dtensor_2d_tp_hsdp_float8_composable"]
        self.assertGreater(t1["peak_reserved_hbm_gb"], 78.0)
        self.assertLess(t4["peak_reserved_hbm_gb"], 62.0)
        self.assertEqual(t4["cuda_malloc_retry_stalls_per_100_steps"], 0)
        self.assertGreater(t4["step_throughput_tokens_per_sec_per_gpu"], t1["step_throughput_tokens_per_sec_per_gpu"])


if __name__ == "__main__":
    unittest.main()
