from __future__ import annotations

import unittest

from experiments.zero_bubble_dualpipe.run import (
    evaluate_post_validation_rollback,
    evaluate_schedule_matrix,
    run_simulation,
    validate_results,
    verify_adamw_post_validation_rollback,
)


class TestZeroBubbleDualPipe(unittest.TestCase):
    def test_schedule_ordering_and_memory(self) -> None:
        rows = evaluate_schedule_matrix(pp_list=[16], mb_list=[64])
        by_name = {r.schedule: r for r in rows}
        self.assertLess(by_name["DualPipe"].bubble_rate_pct, by_name["ZB-H1 (ZB1P)"].bubble_rate_pct)
        self.assertLess(by_name["ZB-H1 (ZB1P)"].bubble_rate_pct, by_name["1F1B"].bubble_rate_pct)
        self.assertEqual(by_name["DualPipe"].peak_in_flight_activations_chunks, 17)
        self.assertEqual(by_name["DualPipe"].param_replication_factor, 2.0)
        self.assertEqual(by_name["DualPipeV"].param_replication_factor, 1.0)
        self.assertLess(by_name["DualPipe"].exposed_comm_pct, 1.0)

    def test_post_validation_adamw_rollback_exactness(self) -> None:
        err = verify_adamw_post_validation_rollback()
        self.assertLess(err, 1e-14)
        rb = evaluate_post_validation_rollback()
        self.assertTrue(all(r.throughput_gain_pct > 3.0 for r in rb))

    def test_full_simulation_validation(self) -> None:
        data = run_simulation()
        validate_results(data)


if __name__ == "__main__":
    unittest.main()
