import unittest
from experiments.cuda_multi_stream_legacy_default_stream_sync_item_nonzero_stall_selection.run import evaluate_cuda_multi_stream_sync_stalls


class TestCudaMultiStreamSyncStalls(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_cuda_multi_stream_sync_stalls()
        self.assertEqual(res["round"], 306)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier4_nonblocking_streams_ptds_device_masking_plus_event_record_stream"]["comm_compute_overlap_pct"],
            90.0,
        )
        self.assertEqual(
            tiers["tier4_nonblocking_streams_ptds_device_masking_plus_event_record_stream"]["gpu_host_sync_bubble_ms"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
