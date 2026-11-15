import unittest
from experiments.llm_ddp_fsdp_gradient_accumulation_no_sync_bucket_cap_selection.run import (
    evaluate_fsdp_gradient_accumulation_no_sync_tiers,
)


class TestFsdpGradientAccumulationNoSyncSelection(unittest.TestCase):
    def test_fsdp_gas_no_sync_tiers(self) -> None:
        data = evaluate_fsdp_gradient_accumulation_no_sync_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_default_fsdp_every_microbatch_reducescatter_comm_storm"]
        t2 = tiers["tier2_blind_fsdp_no_sync_unsharded_140gb_gradient_oom_crash"]
        t4 = tiers["tier4_hsdp_2d_intra_node_nvlink_shard_plus_inter_node_roce_no_sync"]
        self.assertEqual(t2["peak_gradient_hbm_gb"], 140.0)
        self.assertEqual(t4["peak_gradient_hbm_gb"], 17.5)
        self.assertEqual(t4["inter_node_roce_collectives_per_step"], 1)
        self.assertLess(t4["optimizer_step_latency_sec"], t1["optimizer_step_latency_sec"])


if __name__ == "__main__":
    unittest.main()
