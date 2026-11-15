import unittest
from experiments.ai_cluster_k8s_kubelet_gpu_pleg_pod_oomkill_dmesg_eviction_selection.run import evaluate_k8s_pod_cgroup_oomkill


class TestK8sPodCgroupOomkill(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_k8s_pod_cgroup_oomkill()
        self.assertEqual(res["round"], 286)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_buffered_io_mmap_no_fadvise_hard_memory_max"]["pod_working_set_gib"],
            512.0,
        )
        self.assertEqual(
            tiers["tier4_o_direct_fadv_dontneed_plus_memory_high_plus_shm_budget"]["oomkill_or_eviction_rate_pct"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
