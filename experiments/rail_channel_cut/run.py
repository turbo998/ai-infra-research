"""Account multi-channel Ring payload across per-rail NICs and intra-node bottleneck cuts."""

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
    "own-offline synthetic arithmetic; no GPU, NIC, PCIe, NVLink, RDMA or elapsed-time measurement",
    "ideal equal-chunk multi-channel Ring AllReduce; not NCCL backtracking execution or runtime tuning",
    "models per-NIC full-duplex payload ceiling and a single shared intra-node bottleneck edge ceiling",
    "excludes latency, protocol headers, LL/LL128/Simple framing, PXN hop overhead and reduction compute",
    "network and intra-node link payload are not GPU HBM traffic; no quality, SLO, delivery or cost guarantee",
]


def _ascii_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or not value.replace("_", "").isalnum():
        raise ValueError(f"{name}: expected non-empty ASCII alphanumeric/underscore label")
    return value


def rail_channel_cut(
    *,
    case_id: str,
    node_by_rank: list[str],
    channels: list[dict],
    message_bytes: int,
    nic_bytes_per_second: int,
    intra_node_bytes_per_second: int,
) -> dict:
    """Account per-channel directed Ring edges across per-rail NICs and intra-node links."""
    label = _ascii_label(case_id, "case_id")
    if (
        not isinstance(node_by_rank, list)
        or len(node_by_rank) < 2
        or any(not isinstance(node, str) or not node.isascii() or not node.isalnum() for node in node_by_rank)
    ):
        raise ValueError("node_by_rank: expected at least two ranks with ASCII alphanumeric node labels")
    ranks = len(node_by_rank)
    nodes = sorted(set(node_by_rank))
    if not isinstance(channels, list) or not channels or any(not isinstance(ch, dict) for ch in channels):
        raise ValueError("channels: expected non-empty list of channel definitions")
    channel_count = len(channels)
    message = integer(message_bytes, "message_bytes")
    nic_ceiling = integer(nic_bytes_per_second, "nic_bytes_per_second")
    intra_ceiling = integer(intra_node_bytes_per_second, "intra_node_bytes_per_second")
    divisor = ranks * channel_count
    if message % divisor:
        raise ValueError("message_bytes: equal chunks require divisibility by (ranks * channel_count)")

    per_channel_edge = 2 * (ranks - 1) * (message // divisor)
    nic_sent: dict[str, int] = {}
    nic_received: dict[str, int] = {}
    intra_edge_bytes: dict[str, int] = {}
    cross_nic_node_channels = 0
    total_cross_node_channel_edges = 0

    for idx, ch in enumerate(channels):
        ring = ch.get("ring")
        if (
            not isinstance(ring, list)
            or any(type(rank) is not int for rank in ring)
            or sorted(ring) != list(range(ranks))
        ):
            raise ValueError(f"channels[{idx}].ring: expected a permutation of all rank indices")
        send_map = ch.get("send_nic_by_node")
        recv_map = ch.get("recv_nic_by_node")
        if not isinstance(send_map, dict) or not isinstance(recv_map, dict):
            raise ValueError(f"channels[{idx}]: expected send_nic_by_node and recv_nic_by_node dicts")
        if sorted(send_map.keys()) != nodes or sorted(recv_map.keys()) != nodes:
            raise ValueError(f"channels[{idx}]: NIC maps must cover exact node set {nodes}")
        for node in nodes:
            send_nic = _ascii_label(send_map[node], f"channels[{idx}].send_nic_by_node[{node}]")
            recv_nic = _ascii_label(recv_map[node], f"channels[{idx}].recv_nic_by_node[{node}]")
            nic_sent.setdefault(f"{node}:{send_nic}", 0)
            nic_received.setdefault(f"{node}:{recv_nic}", 0)

        node_has_cross = dict.fromkeys(nodes, False)
        for pos, src in enumerate(ring):
            dst = ring[(pos + 1) % ranks]
            src_node = node_by_rank[src]
            dst_node = node_by_rank[dst]
            if src_node != dst_node:
                send_key = f"{src_node}:{send_map[src_node]}"
                recv_key = f"{dst_node}:{recv_map[dst_node]}"
                nic_sent[send_key] += per_channel_edge
                nic_received[recv_key] += per_channel_edge
                node_has_cross[src_node] = True
                node_has_cross[dst_node] = True
                total_cross_node_channel_edges += 1
            else:
                edge_key = f"{src_node}:{src}->{dst}"
                intra_edge_bytes[edge_key] = intra_edge_bytes.get(edge_key, 0) + per_channel_edge

        for node in nodes:
            if node_has_cross[node] and send_map[node] != recv_map[node]:
                cross_nic_node_channels += 1

    max_nic_sent = max(nic_sent.values(), default=0)
    max_nic_received = max(nic_received.values(), default=0)
    max_nic_direction = max(max_nic_sent, max_nic_received)
    max_intra_edge = max(intra_edge_bytes.values(), default=0)

    try:
        nic_time = max_nic_direction / nic_ceiling
        intra_time = max_intra_edge / intra_ceiling
    except OverflowError as exc:
        raise ValueError("rail_channel_cut: derived time is not finite") from exc
    if (
        not math.isfinite(nic_time)
        or not math.isfinite(intra_time)
        or (max_nic_direction > 0 and nic_time <= 0)
        or (max_intra_edge > 0 and intra_time <= 0)
    ):
        raise ValueError("rail_channel_cut: derived nonzero time is not finite and positive")

    combined_time = max(nic_time, intra_time)
    if combined_time == 0:
        bottleneck = "none"
    elif nic_time > intra_time:
        bottleneck = "nic"
    elif intra_time > nic_time:
        bottleneck = "intra_node"
    else:
        bottleneck = "tied"

    return {
        "case_id": label,
        "ranks": ranks,
        "channel_count": channel_count,
        "payload_per_channel_edge_bytes": per_channel_edge,
        "cross_node_channel_directed_edges": total_cross_node_channel_edges,
        "cross_nic_node_channel_assignments": cross_nic_node_channels,
        "nic_sent_bytes": dict(sorted(nic_sent.items())),
        "nic_received_bytes": dict(sorted(nic_received.items())),
        "max_nic_direction_bytes": max_nic_direction,
        "max_intra_node_directed_edge_bytes": max_intra_edge,
        "nic_send_or_recv_ceiling_bytes_per_second": nic_ceiling,
        "intra_node_edge_ceiling_bytes_per_second": intra_ceiling,
        "nic_payload_time_lower_bound_seconds": nic_time,
        "intra_node_payload_time_lower_bound_seconds": intra_time,
        "combined_payload_time_lower_bound_seconds": combined_time,
        "dominant_bottleneck": bottleneck,
    }


def calculate(fixture: dict) -> dict:
    """Evaluate explicitly synthetic multi-channel rail/intra-node cases."""
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
        "cases": [{"inputs": case, "result": rail_channel_cut(**case)} for case in fixture["cases"]],
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
                print("Rail channel cut check failed: missing or different results.json", file=sys.stderr)
                return 1
        else:
            with output.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        print("Rail channel cut failed: invalid fixture or output", file=sys.stderr)
        return 1
    print("own-offline rail channel cut verified" if args.check else "own-offline rail channel cut regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
