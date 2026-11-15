import unittest
from experiments.ai_cluster_k8s_dra_nri_topo_lstopo_nic_gpu_numa_affinity_selection.run import (
    evaluate_k8s_gpu_nic_topology_dra_tiers,
)


class TestK8sGpuNicTopologyDraSelection(unittest.TestCase):
    def test_k8s_dra_tiers(self) -> None:
        data = evaluate_k8s_gpu_nic_topology_dra_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_independent_device_plugins_topology_none"]
        t2 = tiers["tier2_kubelet_topology_manager_single_numa_node_policy"]
        t4 = tiers["tier4_k8s_dra_dynamic_resource_allocation_or_cdi_pix_co_allocation"]
        self.assertLess(t1["effective_gdr_bw_per_nic_gbps"], 12.0)
        self.assertEqual(t2["full_node_8gpu_pod_admission_rate_pct"], 0.0)
        self.assertEqual(t4["effective_gdr_bw_per_nic_gbps"], 47.5)
        self.assertEqual(t4["full_node_8gpu_pod_admission_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
