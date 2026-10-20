import unittest

from experiments.sglang_pd_role_switch_offload_sim.run import (
    RoleSwitchScenario,
    evaluate_pd_role_switch,
    run_simulation,
    simulate_decode_incremental_offload,
)


class TestSGLangPdRoleSwitchOffloadSim(unittest.TestCase):
    def test_role_switch_preconditions_and_headroom(self) -> None:
        ok = evaluate_pd_role_switch(
            RoleSwitchScenario("ok", "prefill", "decode")
        )
        self.assertTrue(ok["success"])

        oom = evaluate_pd_role_switch(
            RoleSwitchScenario(
                "oom",
                "prefill",
                "decode",
                decode_cuda_graph_memory_gb=4.0,
                available_gpu_memory_gb=2.0,
            )
        )
        self.assertFalse(oom["success"])
        self.assertTrue(oom["safe_to_restore"])

        fatal = evaluate_pd_role_switch(
            RoleSwitchScenario(
                "fatal",
                "prefill",
                "decode",
                teardown_or_rebuild_raises=True,
            )
        )
        self.assertFalse(fatal["success"])
        self.assertFalse(fatal["safe_to_restore"])
        self.assertTrue(fatal["marked_unhealthy"])

    def test_incremental_decode_offload_recovers_prompt_tail(self) -> None:
        offload = simulate_decode_incremental_offload(
            origin_prompt_len=230,
            page_size=64,
            offload_stride=128,
            decode_steps=260,
        )
        self.assertEqual(offload["prefill_offloaded_len"], 192)
        self.assertEqual(offload["unaligned_prompt_tail_recovered_by_decode"], 38)
        self.assertEqual(offload["total_offload_triggers"], 2)
        self.assertEqual(offload["offload_events"][0]["chunk_start"], 192)
        self.assertEqual(offload["offload_events"][0]["chunk_end"], 320)

    def test_full_simulation(self) -> None:
        res = run_simulation()
        self.assertTrue(
            res["pp_consensus_store"]["arbitrary_code_execution_blocked"]
        )
        self.assertGreater(
            res["pp_consensus_store"]["wire_size_reduction_pct"], 50.0
        )


if __name__ == "__main__":
    unittest.main()
