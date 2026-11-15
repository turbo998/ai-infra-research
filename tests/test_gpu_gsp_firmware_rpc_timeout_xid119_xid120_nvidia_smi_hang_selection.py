import unittest
from experiments.gpu_gsp_firmware_rpc_timeout_xid119_xid120_nvidia_smi_hang_selection.run import (
    evaluate_gsp_xid119_dstate_hang_tiers,
)


class TestGpuGspXid119DstateHangSelection(unittest.TestCase):
    def test_gsp_xid119_tiers(self) -> None:
        data = evaluate_gsp_xid119_dstate_hang_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unprotected_nvml_polling_480_dstate_zombie_snowball"]
        t4 = tiers["tier4_kmsg_xid119_singleflight_breaker_plus_oob_bmc_power_reset"]
        self.assertGreater(t1["stuck_dstate_processes_count"], 400)
        self.assertEqual(t4["stuck_dstate_processes_count"], 1)
        self.assertFalse(t4["kubelet_pleg_and_ssh_frozen"])
        self.assertLess(t4["node_recovery_time_min"], 3.0)


if __name__ == "__main__":
    unittest.main()
