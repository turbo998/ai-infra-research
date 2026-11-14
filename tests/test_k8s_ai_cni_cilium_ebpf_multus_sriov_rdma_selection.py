import unittest
from experiments.k8s_ai_cni_cilium_ebpf_multus_sriov_rdma_selection.run import (
    evaluate_k8s_ai_cni_tiers,
)


class TestK8sAiCniSelection(unittest.TestCase):
    def test_tiers_and_isolation(self) -> None:
        data = evaluate_k8s_ai_cni_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_hostnetwork_shared_root_netns"]
        t2 = tiers["tier2_single_veth_iptables_overlay_soft_roce"]
        t4 = tiers["tier4_cilium_ebpf_netkit_plus_multus_sriov_exclusive_rdma"]
        self.assertGreater(t1["multi_tenant_port_conflict_rate_pct"], 10.0)
        self.assertLess(t2["rdma_busbw_gbs"], 5.0)
        self.assertGreaterEqual(t4["rdma_busbw_gbs"], 48.0)
        self.assertEqual(t4["multi_tenant_port_conflict_rate_pct"], 0.0)
        self.assertTrue(t4["hardware_rdma_netns_isolation"])


if __name__ == "__main__":
    unittest.main()
