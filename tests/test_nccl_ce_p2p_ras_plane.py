import unittest

from experiments.nccl_ce_p2p_ras_plane.run import _check, run_simulation


class TestNCCLCEP2PRASPlane(unittest.TestCase):
    def test_p2p_ce_and_ras_monitor(self) -> None:
        res = run_simulation()
        _check(res)
        self.assertEqual(
            res["p2p_copy_engine_overlap"]["modes"]["p2p_cumem_ce_memcpy_proxy"]["comm_sms_occupied"],
            0,
        )
        self.assertEqual(res["ras_progress_monitor"]["client_port"], 28028)


if __name__ == "__main__":
    unittest.main()
