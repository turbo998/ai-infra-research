"""Unit tests for the cluster orchestration, topology placement, XID isolation, and elastic goodput evaluator."""

import json
import unittest

from experiments.cluster_orchestration.run import (
    HERE,
    evaluate_cluster_orchestration_case,
    evaluate_fixture,
    evaluate_k8s_topology_merge,
    evaluate_nvml_xid_isolation,
    evaluate_slurm_block_topology,
    main,
)


class ClusterOrchestrationTest(unittest.TestCase):
    def test_fixture_check_passes(self) -> None:
        self.assertEqual(main(["--check"]), 0)

    def test_cases_invariants(self) -> None:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        output = evaluate_fixture(fixture)
        by_id = {item["case_id"]: item for item in output["cases"]}

        c1 = by_id["single_numa_pcie_aligned_vs_cross_numa_8gpu"]
        self.assertTrue(c1["k8s_topology"]["pod_admitted"])
        self.assertEqual(c1["k8s_topology"]["merged_affinity_mask"], 1)
        self.assertEqual(c1["slurm_block"]["best_initial_bblock_index"], 1)
        self.assertEqual(c1["slurm_block"]["best_initial_bblock_avail_nodes"], 16)
        self.assertTrue(c1["nvml_xid_isolation"]["ignored_by_default_policy"])
        self.assertEqual(c1["nvml_xid_isolation"]["unhealthy_mig_slices"], 0)

        c2 = by_id["cross_numa_nic_mismatch_rejected_and_block_rounding"]
        self.assertFalse(c2["k8s_topology"]["pod_admitted"])
        self.assertEqual(c2["k8s_topology"]["merged_affinity_mask"], 0)
        self.assertEqual(c2["slurm_block"]["raw_ceil_log2_level"], 1)
        self.assertEqual(c2["slurm_block"]["selected_block_level"], 2)
        self.assertEqual(c2["slurm_block"]["selected_block_capacity_nodes"], 72)
        self.assertEqual(c2["nvml_xid_isolation"]["blast_radius_scope"], "entire_parent_gpu")
        self.assertEqual(c2["nvml_xid_isolation"]["unhealthy_mig_slices"], 7)

        c3 = by_id["mig_isolated_xid_vs_elastic_hot_standby_256node"]
        self.assertTrue(c3["k8s_topology"]["pod_admitted"])
        self.assertEqual(c3["k8s_topology"]["merged_numa_count"], 2)
        self.assertEqual(c3["nvml_xid_isolation"]["blast_radius_scope"], "single_mig_slice")
        self.assertEqual(c3["nvml_xid_isolation"]["unhealthy_mig_slices"], 1)
        self.assertEqual(c3["nvml_xid_isolation"]["healthy_remaining_mig_slices"], 6)
        self.assertGreater(
            c3["elastic_goodput"]["goodput_hot_standby"],
            c3["elastic_goodput"]["goodput_cold_reprovision"],
        )

    def test_validation_guards(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_k8s_topology_merge(
                numa_nodes=9,
                cpu_hint_mask=1,
                gpu_hint_mask=1,
                rdma_nic_hint_mask=1,
                cpu_hint_preferred=True,
                gpu_hint_preferred=True,
                rdma_nic_hint_preferred=True,
                k8s_policy="single-numa-node",
            )
        with self.assertRaises(ValueError):
            evaluate_slurm_block_topology(
                bblock_node_cnt=16,
                block_levels=[0, 1],
                requested_nodes=8,
                segment_size=4,
                available_nodes_per_bblock=[20],
            )
        with self.assertRaises(ValueError):
            evaluate_nvml_xid_isolation(
                nvml_xid_code=48,
                nvml_gi_id=-1,
                nvml_ci_id=0,
                total_mig_slices_on_gpu=7,
            )


if __name__ == "__main__":
    unittest.main()
