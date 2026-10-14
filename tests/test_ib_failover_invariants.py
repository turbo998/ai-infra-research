import json
from pathlib import Path
import unittest

from experiments.ib_failover_invariants.run import calculate, evaluate_ib_failover_case, main


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "ib_failover_invariants"


class IbFailoverInvariantsTest(unittest.TestCase):
    def test_checked_in_results_match(self) -> None:
        self.assertEqual(main(["--check"]), 0)

    def test_selective_retransmit_and_signed_event_conservation(self) -> None:
        fixture = json.loads((EXP / "input.json").read_text(encoding="utf-8"))
        out = calculate(fixture)
        self.assertEqual(out["evidence_status"], "own-offline")
        self.assertTrue(out["synthetic"])
        by_id = {entry["result"]["case_id"]: entry["result"] for entry in out["cases"]}

        lost_ack = by_id["lost_ack_all_delivered_zero_retransmit"]
        self.assertTrue(lost_ack["failover_permitted"])
        self.assertTrue(lost_ack["zero_retransmit_fast_complete"])
        self.assertEqual(lost_ack["selective_retransmit_bytes"], 0)
        self.assertEqual(lost_ack["avoided_duplicate_retransmit_bytes"], 1048576)

        partial = by_id["partial_loss_selective_retransmit_and_signed_events"]
        self.assertTrue(partial["failover_permitted"])
        self.assertEqual(partial["resiliency_send_cq_size_per_dev"], 512)
        self.assertEqual(partial["resiliency_recv_cq_size_per_dev"], 1024)
        self.assertEqual(partial["resiliency_recv_rq_size_per_qp"], 256)
        self.assertEqual(partial["selective_retransmit_bytes"], 262144)
        self.assertEqual(partial["post_events_by_dev"], [1, -1])
        self.assertTrue(partial["resiliency_request_complete"])
        self.assertFalse(partial["naive_unsigned_complete"])

        dual_fail = by_id["dual_device_failure_fatal_ceiling"]
        self.assertFalse(dual_fail["failover_permitted"])
        self.assertFalse(dual_fail["resiliency_request_complete"])

        single_nic = by_id["single_nic_comm_cannot_failover"]
        self.assertFalse(single_nic["failover_permitted"])

    def test_invalid_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_ib_failover_case(
                case_id="bad_weights",
                ndevs=2,
                nqps=4,
                max_requests=64,
                message_bytes=1048576,
                weights_by_dev=[60, 60],
                qp_dev_by_index=[0, 1, 0, 1],
                failed_dev_indices=[0],
                wc_status="IBV_WC_RETRY_EXC_ERR",
                receiver_completions_by_qp=[True, True, True, True],
                receiver_initial_events_by_dev=[2, 2],
                receiver_delivered_completions_by_dev=[2, 2],
            )


if __name__ == "__main__":
    unittest.main()
