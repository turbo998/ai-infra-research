"""Evaluate NCCL IB resiliency CQ/RQ sizing, selective retransmission, and signed event invariants."""

import argparse
import json
from pathlib import Path
import sys

if __package__:
    from ..capacity_roofline.budget import integer
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.capacity_roofline.budget import integer


HERE = Path(__file__).resolve().parent
IB_WRITE_CHUNK_ALIGNMENT = 128
NCCL_IB_MAX_QPS = 128
RECOVERABLE_WC_STATUSES = {"IBV_WC_WR_FLUSH_ERR", "IBV_WC_RETRY_EXC_ERR"}
LIMITATIONS = [
    "own-offline synthetic arithmetic; no IB HCA, RDMA switch, PCIe, GPU or NCCL runtime execution",
    "models static invariants in NVIDIA/nccl@3e46c089f4ae4732722a419eb6096aead7e47a30 net_ib resiliency",
    "does not simulate second-failure race during retransmission (noted as TODO in p2p_resiliency.cc:219-225)",
    "excludes UC alive-message wire loss, PFC pause dynamics, and OS thread scheduling jitter",
]


def _ascii_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or not value.replace("_", "").isalnum():
        raise ValueError(f"{name}: expected non-empty ASCII alphanumeric/underscore label")
    return value


def _divup(a: int, b: int) -> int:
    return (a + b - 1) // b


def evaluate_ib_failover_case(
    *,
    case_id: str,
    ndevs: int,
    nqps: int,
    max_requests: int,
    message_bytes: int,
    weights_by_dev: list[int],
    qp_dev_by_index: list[int],
    failed_dev_indices: list[int],
    wc_status: str,
    receiver_completions_by_qp: list[bool],
    receiver_initial_events_by_dev: list[int],
    receiver_delivered_completions_by_dev: list[int],
) -> dict:
    """Evaluate CQ/RQ sizing, fatal error gating, selective retransmission, and signed events."""
    label = _ascii_label(case_id, "case_id")
    dev_count = integer(ndevs, "ndevs")
    qp_count = integer(nqps, "nqps")
    req_limit = integer(max_requests, "max_requests")
    msg_bytes = integer(message_bytes, "message_bytes")
    status = _ascii_label(wc_status, "wc_status")

    if dev_count > 4 or qp_count > NCCL_IB_MAX_QPS:
        raise ValueError("ndevs or nqps exceeds NCCL IB limits")
    if (
        not isinstance(weights_by_dev, list)
        or len(weights_by_dev) != dev_count
        or any(type(w) is not int or w <= 0 for w in weights_by_dev)
        or sum(weights_by_dev) != 100
    ):
        raise ValueError("weights_by_dev: expected positive integer percentages summing to 100")
    if (
        not isinstance(qp_dev_by_index, list)
        or len(qp_dev_by_index) != qp_count
        or any(type(d) is not int or d < 0 or d >= dev_count for d in qp_dev_by_index)
    ):
        raise ValueError("qp_dev_by_index: expected valid device index per QP")
    if (
        not isinstance(failed_dev_indices, list)
        or any(type(d) is not int or d < 0 or d >= dev_count for d in failed_dev_indices)
        or len(set(failed_dev_indices)) != len(failed_dev_indices)
    ):
        raise ValueError("failed_dev_indices: expected distinct valid device indices")
    if (
        not isinstance(receiver_completions_by_qp, list)
        or len(receiver_completions_by_qp) != qp_count
        or any(type(flag) is not bool for flag in receiver_completions_by_qp)
    ):
        raise ValueError("receiver_completions_by_qp: expected boolean list of length nqps")
    if (
        not isinstance(receiver_initial_events_by_dev, list)
        or len(receiver_initial_events_by_dev) != dev_count
        or any(type(v) is not int or v < 0 for v in receiver_initial_events_by_dev)
    ):
        raise ValueError("receiver_initial_events_by_dev: expected non-negative integer list of length ndevs")
    if (
        not isinstance(receiver_delivered_completions_by_dev, list)
        or len(receiver_delivered_completions_by_dev) != dev_count
        or any(type(v) is not int or v < 0 for v in receiver_delivered_completions_by_dev)
    ):
        raise ValueError("receiver_delivered_completions_by_dev: expected non-negative integer list of length ndevs")

    qps_per_dev = [0] * dev_count
    for d in qp_dev_by_index:
        qps_per_dev[d] += 1
    if any(c == 0 for c in qps_per_dev):
        raise ValueError("qp_dev_by_index: every device in comm must own at least one QP")

    # 1. CQ & RQ sizing invariants (ncclIbResiliencyDataCqSizeGet / DataRqSizeGet)
    baseline_send_cq_per_dev = req_limit * qp_count
    baseline_recv_cq_per_dev = req_limit * 2 * qp_count
    resiliency_send_cq_per_dev = baseline_send_cq_per_dev * dev_count
    resiliency_recv_cq_per_dev = baseline_recv_cq_per_dev * dev_count
    resiliency_recv_rq_per_qp = req_limit * qp_count

    # 2. Fatal vs recoverable gate (ncclIbResiliencyCheckErrorNotFatal)
    n_failed_devices = len(failed_dev_indices)
    recoverable_status = status in RECOVERABLE_WC_STATUSES
    not_fatal = recoverable_status and (n_failed_devices <= 1) and (n_failed_devices < dev_count)

    # 3. Per-QP chunking (ncclIbMultiSend equal/weighted 128B-aligned chunking)
    qp_chunk_bytes: list[int] = []
    send_offset = 0
    for qp_idx in range(qp_count):
        orig_dev = qp_dev_by_index[qp_idx]
        raw_chunk = _divup(
            _divup(msg_bytes * weights_by_dev[orig_dev], 100 * qps_per_dev[orig_dev]),
            IB_WRITE_CHUNK_ALIGNMENT,
        ) * IB_WRITE_CHUNK_ALIGNMENT
        chunk = min(msg_bytes - send_offset, raw_chunk)
        qp_chunk_bytes.append(chunk)
        send_offset = min(msg_bytes, send_offset + chunk)

    # 4. Active QP replacement ring search (ncclIbResiliencyReplaceQps)
    failed_set = set(failed_dev_indices)
    active_qp_target_dev: list[int | None] = []
    for qp_idx in range(qp_count):
        orig_dev = qp_dev_by_index[qp_idx]
        if orig_dev not in failed_set:
            active_qp_target_dev.append(orig_dev)
            continue
        replacement_dev = None
        for offset in range(1, qp_count):
            cand_idx = (qp_idx + offset) % qp_count
            cand_dev = qp_dev_by_index[cand_idx]
            if cand_dev not in failed_set:
                replacement_dev = cand_dev
                break
        active_qp_target_dev.append(replacement_dev)

    # 5. Selective retransmission via RDMA_READ probing (ncclIbResiliencyHandleProbeCompleted)
    rdma_read_probe_bytes = NCCL_IB_MAX_QPS * 1  # sizeof(bool) * NCCL_IB_MAX_QPS = 128 B
    blind_retransmit_bytes = sum(qp_chunk_bytes) if not_fatal else 0
    selective_retransmit_bytes = (
        sum(qp_chunk_bytes[q] for q in range(qp_count) if not receiver_completions_by_qp[q])
        if not_fatal
        else 0
    )
    retransmitted_qp_count = (
        sum(1 for q in range(qp_count) if not receiver_completions_by_qp[q])
        if not_fatal
        else 0
    )
    zero_retransmit_fast_complete = not_fatal and (retransmitted_qp_count == 0)

    # 6. Signed event counter conservation (ncclIbResiliencyRequestIsComplete)
    post_events_by_dev = [
        receiver_initial_events_by_dev[d] - receiver_delivered_completions_by_dev[d]
        for d in range(dev_count)
    ]
    remaining_events_sum = sum(e for e in post_events_by_dev if e >= 0)
    negative_events_sum = sum(e for e in post_events_by_dev if e < 0)
    resiliency_request_complete = (remaining_events_sum == 0) or (
        abs(negative_events_sum) == remaining_events_sum
    )
    naive_unsigned_complete = all(e == 0 for e in post_events_by_dev)

    return {
        "case_id": label,
        "ndevs": dev_count,
        "nqps": qp_count,
        "failed_device_count": n_failed_devices,
        "recoverable_wc_status": recoverable_status,
        "failover_permitted": not_fatal,
        "baseline_send_cq_size_per_dev": baseline_send_cq_per_dev,
        "resiliency_send_cq_size_per_dev": resiliency_send_cq_per_dev,
        "baseline_recv_cq_size_per_dev": baseline_recv_cq_per_dev,
        "resiliency_recv_cq_size_per_dev": resiliency_recv_cq_per_dev,
        "resiliency_recv_rq_size_per_qp": resiliency_recv_rq_per_qp,
        "qp_chunk_bytes": qp_chunk_bytes,
        "active_qp_target_dev": active_qp_target_dev,
        "rdma_read_probe_bytes": rdma_read_probe_bytes,
        "blind_retransmit_bytes": blind_retransmit_bytes,
        "selective_retransmit_bytes": selective_retransmit_bytes,
        "avoided_duplicate_retransmit_bytes": blind_retransmit_bytes - selective_retransmit_bytes,
        "retransmitted_qp_count": retransmitted_qp_count,
        "zero_retransmit_fast_complete": zero_retransmit_fast_complete,
        "post_events_by_dev": post_events_by_dev,
        "remaining_events_sum": remaining_events_sum,
        "negative_events_sum": negative_events_sum,
        "resiliency_request_complete": resiliency_request_complete,
        "naive_unsigned_complete": naive_unsigned_complete,
    }


def calculate(fixture: dict) -> dict:
    """Evaluate all synthetic NCCL IB resiliency invariant cases."""
    if (
        not isinstance(fixture, dict)
        or type(fixture.get("schema_version")) is not int
        or fixture["schema_version"] != 1
        or fixture.get("synthetic") is not True
        or not isinstance(fixture.get("cases"), list)
        or not fixture["cases"]
        or any(not isinstance(case, dict) for case in fixture["cases"])
    ):
        raise ValueError("fixture: expected version-one nonempty synthetic cases")
    return {
        "schema_version": 1,
        "evidence_status": "own-offline",
        "synthetic": True,
        "cases": [{"inputs": case, "result": evaluate_ib_failover_case(**case)} for case in fixture["cases"]],
        "limitations": LIMITATIONS,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate results.json or verify exact match with --check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        content = json.dumps(calculate(fixture), indent=2, sort_keys=True, allow_nan=False) + "\n"
        output = HERE / "results.json"
        if args.check:
            if not output.is_file() or output.read_bytes() != content.encode("utf-8"):
                print("IB failover invariants check failed: missing or different results.json", file=sys.stderr)
                return 1
        else:
            with output.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        print("IB failover invariants failed: invalid fixture or output", file=sys.stderr)
        return 1
    print("own-offline ib failover invariants verified" if args.check else "own-offline ib failover invariants regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
