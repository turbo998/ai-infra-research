"""Count ideal directed Ring payload across node cuts, never benchmark a network."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__:
    from ..capacity_roofline.budget import integer
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.capacity_roofline.budget import integer


HERE = Path(__file__).resolve().parent
LIMITATIONS = [
    "own-offline synthetic arithmetic; no GPU, NIC, RDMA or elapsed-time measurement",
    "one ideal equal-chunk Ring AllReduce; not NCCL topology search or algorithm selection",
    "one shared send-side NIC payload ceiling per node, with independent full-duplex receive",
    "excludes latency, protocol headers, padding, channels, contention and reduction compute",
    "network payload is not GPU HBM traffic; no quality, SLO, delivery or cost guarantee",
]


def ring_cut(
    *, node_by_rank: list[str], ring: list[int], message_bytes: int,
    nic_bytes_per_second: int,
) -> dict:
    """Account per directed edge and per node's shared outgoing NIC."""
    if (
        not isinstance(node_by_rank, list) or len(node_by_rank) < 2
        or any(not isinstance(node, str) or not node.isascii() or not node.isalnum()
               for node in node_by_rank)
    ):
        raise ValueError("node_by_rank: expected at least two ranks with synthetic node labels")
    ranks = len(node_by_rank)
    if (
        not isinstance(ring, list) or any(type(rank) is not int for rank in ring)
        or sorted(ring) != list(range(ranks))
    ):
        raise ValueError("ring: expected a permutation of all rank indices")
    message = integer(message_bytes, "message_bytes")
    ceiling = integer(nic_bytes_per_second, "nic_bytes_per_second")
    if message % ranks:
        raise ValueError("message_bytes: equal chunks require divisibility by rank count")
    per_edge = 2 * (ranks - 1) * (message // ranks)
    outgoing = dict.fromkeys(sorted(set(node_by_rank)), 0)
    for position, rank in enumerate(ring):
        peer = ring[(position + 1) % ranks]
        if node_by_rank[rank] != node_by_rank[peer]:
            outgoing[node_by_rank[rank]] += 1
    sent = {node: cuts * per_edge for node, cuts in outgoing.items()}
    try:
        time_floor = max(sent.values()) / ceiling
    except OverflowError as exc:
        raise ValueError("ring_cut: derived time is not finite") from exc
    if not math.isfinite(time_floor) or (max(sent.values()) > 0 and time_floor <= 0):
        raise ValueError("ring_cut: derived nonzero time is not finite and positive")
    return {
        "ranks": ranks,
        "payload_per_directed_edge_bytes": per_edge,
        "cross_node_directed_edges": sum(outgoing.values()),
        "outgoing_cross_edges_by_node": outgoing,
        "node_network_sent_bytes": sent,
        "fabric_network_sent_bytes": sum(sent.values()),
        "node_send_payload_ceiling_bytes_per_second": ceiling,
        "nic_payload_time_lower_bound_seconds": time_floor,
    }


def calculate(fixture: dict) -> dict:
    """Evaluate explicitly synthetic cases with reproducible inputs and limitations."""
    if (
        not isinstance(fixture, dict) or type(fixture.get("schema_version")) is not int
        or fixture["schema_version"] != 1 or fixture.get("synthetic") is not True
        or not isinstance(fixture.get("cases"), list) or not fixture["cases"]
        or any(not isinstance(case, dict) for case in fixture["cases"])
    ):
        raise ValueError("fixture: expected version-one nonempty synthetic cases")
    return {
        "schema_version": 1, "evidence_status": "own-offline", "synthetic": True,
        "cases": [
            {"inputs": case, "result": ring_cut(**case)} for case in fixture["cases"]
        ],
        "limitations": LIMITATIONS,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate results, or compare exact output bytes without modification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        content = json.dumps(calculate(fixture), indent=2, sort_keys=True, allow_nan=False) + "\n"
        output = HERE / "results.json"
        if args.check:
            if not output.is_file() or output.read_bytes() != content.encode("utf-8"):
                print("Ring cut check failed: missing or different results.json", file=sys.stderr)
                return 1
        else:
            with output.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        print("Ring cut failed: invalid fixture or output", file=sys.stderr)
        return 1
    print("own-offline Ring cut verified" if args.check else "own-offline Ring cut regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
