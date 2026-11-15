import unittest
from experiments.ai_cluster_nccl_socket_ifname_gid_index_rocev2_dscp_fallback_selection.run import evaluate_nccl_rocev2_gid_tc_config


class TestNcclRocev2GidTcConfig(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_nccl_rocev2_gid_tc_config()
        self.assertEqual(res["round"], 292)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier4_auto_rocev2_gid_discovery_plus_tc106_cma_tos_plus_ifname_filter"]["allreduce_busbw_gbps"],
            tiers["tier2_gid3_rocev2_without_nccl_ib_tc_lossy_queue0"]["allreduce_busbw_gbps"] * 15.0,
        )
        self.assertEqual(
            tiers["tier4_auto_rocev2_gid_discovery_plus_tc106_cma_tos_plus_ifname_filter"]["retransmit_or_drop_rate_pct"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
