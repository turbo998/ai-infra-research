import unittest
from experiments.pytorch_elastic_torchrun_rdzv_tcpstore_barrier_zombie_recovery_selection.run import evaluate_torchrun_rdzv_tcpstore_recovery


class TestTorchrunRdzvTcpstoreRecovery(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_torchrun_rdzv_tcpstore_recovery()
        self.assertEqual(res["round"], 303)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_in_process_abort_comm_shrink_plus_hierarchical_store_no_respawn"]["fault_recovery_reinit_sec"],
            10.0,
        )
        self.assertEqual(
            tiers["tier4_in_process_abort_comm_shrink_plus_hierarchical_store_no_respawn"]["restart_oom_failure_rate_pct"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
