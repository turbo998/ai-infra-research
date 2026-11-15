import unittest
from experiments.llm_tensor_parallel_allreduce_vs_sequence_parallel_ag_rs_selection.run import (
    evaluate_tp_sp_async_overlap_tiers,
)


class TestTpSpAsyncOverlapSelection(unittest.TestCase):
    def test_tp_sp_tiers(self) -> None:
        data = evaluate_tp_sp_async_overlap_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_pure_tp_allreduce_replicated_layernorm_and_input_activations"]
        t4 = tiers["tier4_tp_plus_sp_with_te_userbuffers_async_tp_pipelined_overlap"]
        self.assertEqual(t1["activation_hbm_per_gpu_gb"], 64.0)
        self.assertEqual(t4["activation_hbm_per_gpu_gb"], 19.2)
        self.assertEqual(t4["extra_comm_bytes_pct"], 0.0)
        self.assertLess(t4["step_time_ms"], t1["step_time_ms"])


if __name__ == "__main__":
    unittest.main()
