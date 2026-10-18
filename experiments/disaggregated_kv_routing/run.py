#!/usr/bin/env python3
"""CPU/offline analytical and discrete-event model for Disaggregated Prefill/Decode KV Routing.

Covers primary source mechanisms from:
1. `kvcache-ai/Mooncake` (`7ff6bf7f72f15aba7289bc25c87cf491e04efc6f`):
   - `slice.h:69-90` (`planRdmaSlices` with tail-folding `merge_ratio = 0.25`)
   - `deadline_mlu.h:56-66` (`DeadlineMlu(bytes_ahead, length, deadline_ns, now_ns, bw_bps)`)
   - `bw_arbitration.h:77-116` (`OrderByUrgency` greedy slot-by-slot MLU scheduling across contending KV flows)
   - `rail_monitor.h:125-147` (`RailMonitor` circuit breaker with `error_threshold=3`, `cooldown=30s`, half-open probe)
2. `ai-dynamo/nixl` (`c40f1b29b4989765385059196fadcc7bedc37ffb`):
   - `nixl.h:201-237` & `nixl_descriptors.h:148-192` (`nixlStrideDesc` compressed strided KV block descriptor vs
     expanded `nixlBasicDesc` per-layer/per-page scatter-gather descriptor overhead)
3. `sgl-project/sglang` (`4ec19ca27f08f7ce44bbe1da70ce964731940eb2`):
   - `decode_hicache_mixin.py:26-51, 166-181` (`DecodePrefixMatch` L1 HBM + L2 Host DRAM + L3 Storage prefix hits,
     and `HiCacheRestoreGatedKVReceiver` overlapping remote RDMA KV transfer with local L2->L1 PCIe `load_back`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input.json"
DEFAULT_JSON = ROOT / "results.json"
DEFAULT_CSV = ROOT / "results.csv"


def _validate_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _validate_positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return val


def plan_rdma_slices(
    length: int,
    base_block: int,
    max_slices: int,
    merge_ratio: float = 0.25,
) -> dict[str, int]:
    """Exact port of `mooncake::tent::planRdmaSlices` (`slice.h:69-90`)."""
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    if base_block <= 0:
        base_block = 1
    if max_slices <= 0:
        max_slices = 1
    if length == 0:
        return {"block_size": base_block, "count": 1, "last_slice_bytes": 0}

    count = (length + base_block - 1) // base_block
    if count > max_slices:
        count = max_slices

    per_slice = (length + count - 1) // count
    if per_slice % base_block == 0:
        block_size = per_slice
    else:
        block_size = (per_slice // base_block + 1) * base_block

    count = (length + block_size - 1) // block_size
    if count > 1:
        tail = length - (count - 1) * block_size
        if float(tail) < merge_ratio * block_size:
            count -= 1

    last_slice_bytes = length - (count - 1) * block_size
    return {
        "block_size": int(block_size),
        "count": int(count),
        "last_slice_bytes": int(last_slice_bytes),
    }


def deadline_mlu(
    bytes_ahead: int,
    length: int,
    deadline_ns: int,
    now_ns: int,
    bw_bps: float,
) -> float:
    """Exact port of `mooncake::tent::DeadlineMlu` (`deadline_mlu.h:56-66`)."""
    if deadline_ns == 0 or bw_bps <= 0.0:
        return 0.0
    if deadline_ns <= now_ns:
        return float("inf")
    window_s = (deadline_ns - now_ns) / 1e9
    predicted_time_s = float(bytes_ahead + length) / bw_bps
    return predicted_time_s / window_s


def order_by_urgency(
    flows: list[dict[str, int]],
    now_ns: int,
    bw_bps: float,
    bytes_ahead: int = 0,
    greedy_slots: int = 64,
) -> list[int]:
    """Exact port of `mooncake::tent::OrderByUrgency` (`bw_arbitration.h:77-116`)."""
    n = len(flows)
    order: list[int] = []
    taken = [False] * n
    cur_ahead = bytes_ahead

    greedy = min(n, greedy_slots)
    for _ in range(greedy):
        best = n
        best_mlu = 0.0
        for i in range(n):
            if taken[i]:
                continue
            mlu = deadline_mlu(
                cur_ahead,
                int(flows[i]["length"]),
                int(flows[i]["deadline_ns"]),
                now_ns,
                bw_bps,
            )
            if best == n or mlu > best_mlu:
                best = i
                best_mlu = mlu
        taken[best] = True
        order.append(best)
        cur_ahead += int(flows[best]["length"])

    if len(order) == n:
        return order

    rest = [i for i in range(n) if not taken[i]]
    mlu_map = {
        i: deadline_mlu(
            cur_ahead,
            int(flows[i]["length"]),
            int(flows[i]["deadline_ns"]),
            now_ns,
            bw_bps,
        )
        for i in rest
    }
    rest.sort(key=lambda idx: mlu_map[idx], reverse=True)
    order.extend(rest)
    return order


def evaluate_kv_layout_and_slicing(case: dict[str, Any]) -> dict[str, Any]:
    """Compare GQA vs MLA KV footprint, NIXL `nixlStrideDesc` vs `nixlBasicDesc`, and Mooncake `planRdmaSlices`."""
    seq_len = _validate_positive_int("seq_len", case.get("seq_len"))
    page_tokens = _validate_positive_int("page_tokens", case.get("page_tokens"))
    num_layers = _validate_positive_int("num_layers", case.get("num_layers"))
    tp_size = _validate_positive_int("tp_size", case.get("tp_size"))
    kv_bytes_per_token_per_layer = _validate_positive_int(
        "kv_bytes_per_token_per_layer", case.get("kv_bytes_per_token_per_layer")
    )
    base_block_bytes = _validate_positive_int("base_block_bytes", case.get("base_block_bytes", 65536))
    max_slices = _validate_positive_int("max_slices", case.get("max_slices", 64))
    merge_ratio = float(case.get("merge_ratio", 0.25))
    prep_cost_per_desc_ns = _validate_positive_float(
        "prep_cost_per_desc_ns", case.get("prep_cost_per_desc_ns", 85.0)
    )

    num_pages = (seq_len + page_tokens - 1) // page_tokens
    per_rank_bytes_per_token_layer = kv_bytes_per_token_per_layer // tp_size
    if per_rank_bytes_per_token_layer <= 0:
        raise ValueError("kv_bytes_per_token_per_layer // tp_size must be > 0")

    total_kv_bytes_per_rank = seq_len * num_layers * per_rank_bytes_per_token_layer

    # NIXL (`nixl_descriptors.h:32-192`):
    # Expanded `nixlBasicDesc` materializes one descriptor per (layer, page) block (24 bytes each: addr, len, devId)
    # Strided `nixlStrideDesc` compresses regular layer strides per page (or regular page runs per layer) into
    # 1 descriptor per layer (40 bytes each: addr, len, devId, stride, count)
    expanded_basic_desc_count = num_layers * num_pages
    strided_desc_count = num_layers
    expanded_prep_us = (expanded_basic_desc_count * prep_cost_per_desc_ns) / 1000.0
    strided_prep_us = (strided_desc_count * prep_cost_per_desc_ns) / 1000.0

    slice_plan = plan_rdma_slices(
        length=total_kv_bytes_per_rank,
        base_block=base_block_bytes,
        max_slices=max_slices,
        merge_ratio=merge_ratio,
    )
    naive_slice_count = (total_kv_bytes_per_rank + slice_plan["block_size"] - 1) // slice_plan["block_size"]

    return {
        "case_id": str(case["case_id"]),
        "architecture": str(case["architecture"]),
        "seq_len": seq_len,
        "page_tokens": page_tokens,
        "num_pages": num_pages,
        "num_layers": num_layers,
        "tp_size": tp_size,
        "total_kv_bytes_per_rank": total_kv_bytes_per_rank,
        "total_kv_mib_per_rank": round(total_kv_bytes_per_rank / (1024.0 * 1024.0), 3),
        "nixl_expanded_basic_desc_count": expanded_basic_desc_count,
        "nixl_strided_desc_count": strided_desc_count,
        "nixl_desc_reduction_x": round(expanded_basic_desc_count / float(strided_desc_count), 2),
        "nixl_expanded_prep_us": round(expanded_prep_us, 2),
        "nixl_strided_prep_us": round(strided_prep_us, 2),
        "mooncake_slice_block_bytes": slice_plan["block_size"],
        "mooncake_slice_count": slice_plan["count"],
        "mooncake_last_slice_bytes": slice_plan["last_slice_bytes"],
        "mooncake_tail_folded": int(naive_slice_count - slice_plan["count"]),
    }


def evaluate_deadline_arbitration(case: dict[str, Any]) -> dict[str, Any]:
    """Compare FIFO/Equal-Split vs Mooncake `OrderByUrgency` (`bw_arbitration.h` & `deadline_mlu.h`)."""
    nic_bw_gbps = _validate_positive_float("nic_bw_gbps", case.get("nic_bw_gbps"))
    now_ns = _validate_non_negative_int("now_ns", case.get("now_ns", 0))
    bytes_ahead = _validate_non_negative_int("bytes_ahead", case.get("bytes_ahead", 0))
    flows = case.get("flows")
    if not isinstance(flows, list) or not flows:
        raise ValueError("flows must be a non-empty list")

    # Convert Gbps (bits/sec) to bytes/sec for `DeadlineMlu` (`bw_bps` in Mooncake is bytes/sec)
    bw_bytes_per_sec = (nic_bw_gbps * 1e9) / 8.0

    # 1. FIFO order (0, 1, ..., n-1)
    fifo_misses = 0
    fifo_max_mlu = 0.0
    cur_bytes = bytes_ahead
    for idx in range(len(flows)):
        f = flows[idx]
        length = _validate_positive_int(f"flows[{idx}].length", f.get("length"))
        dl_ns = _validate_positive_int(f"flows[{idx}].deadline_ns", f.get("deadline_ns"))
        mlu = deadline_mlu(cur_bytes, length, dl_ns, now_ns, bw_bytes_per_sec)
        if mlu > 1.0:
            fifo_misses += 1
        if mlu > fifo_max_mlu:
            fifo_max_mlu = mlu
        cur_bytes += length

    # 2. Equal-Share Contention (all N flows split NIC bandwidth equally: bw / N)
    equal_split_misses = 0
    per_flow_bw = bw_bytes_per_sec / float(len(flows))
    for f in flows:
        length = int(f["length"])
        dl_ns = int(f["deadline_ns"])
        # Each flow waits for its share of hardware backlog + its own bytes at 1/N bandwidth
        mlu_eq = deadline_mlu(bytes_ahead // len(flows), length, dl_ns, now_ns, per_flow_bw)
        if mlu_eq > 1.0:
            equal_split_misses += 1

    # 3. Mooncake `OrderByUrgency` (`bw_arbitration.h:77-116`)
    urgent_order = order_by_urgency(flows, now_ns=now_ns, bw_bps=bw_bytes_per_sec, bytes_ahead=bytes_ahead)
    urgent_misses = 0
    urgent_max_mlu = 0.0
    cur_bytes = bytes_ahead
    ordered_flow_ids: list[str] = []
    for idx in urgent_order:
        f = flows[idx]
        ordered_flow_ids.append(str(f["flow_id"]))
        length = int(f["length"])
        dl_ns = int(f["deadline_ns"])
        mlu = deadline_mlu(cur_bytes, length, dl_ns, now_ns, bw_bytes_per_sec)
        if mlu > 1.0:
            urgent_misses += 1
        if mlu > urgent_max_mlu:
            urgent_max_mlu = mlu
        cur_bytes += length

    return {
        "case_id": str(case["case_id"]),
        "nic_bw_gbps": nic_bw_gbps,
        "num_contending_flows": len(flows),
        "fifo_slo_misses": fifo_misses,
        "equal_split_slo_misses": equal_split_misses,
        "mooncake_urgent_slo_misses": urgent_misses,
        "fifo_max_mlu": round(fifo_max_mlu, 4),
        "mooncake_urgent_max_mlu": round(urgent_max_mlu, 4),
        "ordered_flow_ids": ordered_flow_ids,
    }


def evaluate_hicache_pd_pipeline(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate SGLang `DecodePrefixMatch` + `HiCacheRestoreGatedKVReceiver` + Mooncake `RailMonitor`."""
    total_prompt_tokens = _validate_positive_int("total_prompt_tokens", case.get("total_prompt_tokens"))
    l1_device_hit_tokens = _validate_non_negative_int("l1_device_hit_tokens", case.get("l1_device_hit_tokens"))
    l2_host_hit_tokens = _validate_non_negative_int("l2_host_hit_tokens", case.get("l2_host_hit_tokens"))
    l3_storage_hit_tokens = _validate_non_negative_int("l3_storage_hit_tokens", case.get("l3_storage_hit_tokens"))
    if l1_device_hit_tokens + l2_host_hit_tokens + l3_storage_hit_tokens > total_prompt_tokens:
        raise ValueError("Sum of L1 + L2 + L3 hit tokens cannot exceed total_prompt_tokens")

    kv_bytes_per_token = _validate_positive_int("kv_bytes_per_token", case.get("kv_bytes_per_token"))
    prefill_compute_tokens_per_ms = _validate_positive_float(
        "prefill_compute_tokens_per_ms", case.get("prefill_compute_tokens_per_ms")
    )
    rdma_bw_gbps = _validate_positive_float("rdma_bw_gbps", case.get("rdma_bw_gbps"))
    l2_pcie_h2d_bw_gbps = _validate_positive_float("l2_pcie_h2d_bw_gbps", case.get("l2_pcie_h2d_bw_gbps"))
    l3_storage_bw_gbps = _validate_positive_float("l3_storage_bw_gbps", case.get("l3_storage_bw_gbps"))
    failed_rails = _validate_non_negative_int("failed_rails", case.get("failed_rails", 0))
    total_rails = _validate_positive_int("total_rails", case.get("total_rails", 8))
    if failed_rails >= total_rails:
        raise ValueError("failed_rails must be < total_rails")

    # `DecodePrefixMatch` properties (`decode_hicache_mixin.py:35-51`):
    decode_prefix_len = l1_device_hit_tokens + l2_host_hit_tokens + l3_storage_hit_tokens
    restore_token_count = decode_prefix_len - l1_device_hit_tokens
    remote_prefill_delta_tokens = total_prompt_tokens - decode_prefix_len

    # Baseline without Decode HiCache (only L1 HBM hits count on Decode; L2/L3 missed):
    no_hicache_remote_tokens = total_prompt_tokens - l1_device_hit_tokens
    no_hicache_prefill_ms = no_hicache_remote_tokens / prefill_compute_tokens_per_ms
    active_rdma_bw_gbps = rdma_bw_gbps * ((total_rails - failed_rails) / float(total_rails))
    no_hicache_rdma_ms = (no_hicache_remote_tokens * kv_bytes_per_token * 8.0) / (active_rdma_bw_gbps * 1e6)
    no_hicache_ttft_ms = no_hicache_prefill_ms + no_hicache_rdma_ms

    # With SGLang Decode HiCache (`DecodeHiCachePreallocMixin` + `HiCacheRestoreGatedKVReceiver`):
    # 1. Remote Prefill computes and RDMA-transfers only `remote_prefill_delta_tokens` (after `decode_prefix_len`).
    remote_prefill_ms = remote_prefill_delta_tokens / prefill_compute_tokens_per_ms
    remote_rdma_ms = (remote_prefill_delta_tokens * kv_bytes_per_token * 8.0) / (active_rdma_bw_gbps * 1e6)
    remote_branch_ms = remote_prefill_ms + remote_rdma_ms

    # 2. Local Decode HiCache branch runs concurrently (`_start_hicache_prefetch` L3->L2 + `_try_hicache_queue_load_back` L2->L1):
    l3_prefetch_ms = (l3_storage_hit_tokens * kv_bytes_per_token * 8.0) / (l3_storage_bw_gbps * 1e6)
    l2_load_back_ms = (restore_token_count * kv_bytes_per_token * 8.0) / (l2_pcie_h2d_bw_gbps * 1e6)
    local_restore_branch_ms = l3_prefetch_ms + l2_load_back_ms

    # `HiCacheRestoreGatedKVReceiver.poll()` (`decode_hicache_mixin.py:172-180`) gates readiness on BOTH:
    # `kv_receiver.poll() == KVPoll.Success` AND `hicache_restore_status == HiCacheRestoreResult.READY`.
    gated_ttft_ms = max(remote_branch_ms, local_restore_branch_ms)
    bottleneck = "remote_prefill_rdma" if remote_branch_ms >= local_restore_branch_ms else "local_hicache_restore"

    return {
        "case_id": str(case["case_id"]),
        "total_prompt_tokens": total_prompt_tokens,
        "decode_prefix_len": decode_prefix_len,
        "restore_token_count": restore_token_count,
        "remote_prefill_delta_tokens": remote_prefill_delta_tokens,
        "active_rdma_bw_gbps": round(active_rdma_bw_gbps, 2),
        "no_hicache_ttft_ms": round(no_hicache_ttft_ms, 2),
        "remote_branch_ms": round(remote_branch_ms, 2),
        "local_restore_branch_ms": round(local_restore_branch_ms, 2),
        "hicache_gated_ttft_ms": round(gated_ttft_ms, 2),
        "ttft_speedup_x": round(no_hicache_ttft_ms / gated_ttft_ms, 2),
        "rdma_traffic_saved_pct": round(
            (no_hicache_remote_tokens - remote_prefill_delta_tokens) / float(max(1, no_hicache_remote_tokens)) * 100.0,
            2,
        ),
        "bottleneck": bottleneck,
    }


def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise ValueError("Fixture schema_version must be 1")

    layout_cases = fixture.get("kv_layout_and_slicing_cases")
    arb_cases = fixture.get("deadline_arbitration_cases")
    hicache_cases = fixture.get("hicache_pd_pipeline_cases")
    if not isinstance(layout_cases, list) or not layout_cases:
        raise ValueError("kv_layout_and_slicing_cases must be a non-empty list")
    if not isinstance(arb_cases, list) or not arb_cases:
        raise ValueError("deadline_arbitration_cases must be a non-empty list")
    if not isinstance(hicache_cases, list) or not hicache_cases:
        raise ValueError("hicache_pd_pipeline_cases must be a non-empty list")

    return {
        "schema_version": 1,
        "experiment_id": "disaggregated_kv_routing",
        "evidence_type": "own-offline",
        "kv_layout_and_slicing": [evaluate_kv_layout_and_slicing(c) for c in layout_cases],
        "deadline_arbitration": [evaluate_deadline_arbitration(c) for c in arb_cases],
        "hicache_pd_pipeline": [evaluate_hicache_pd_pipeline(c) for c in hicache_cases],
    }


def _to_csv_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results["kv_layout_and_slicing"]:
        rows.append(
            {
                "section": "kv_layout_and_slicing",
                "case_id": item["case_id"],
                "primary_metric": item["total_kv_mib_per_rank"],
                "secondary_metric": item["mooncake_slice_count"],
                "delta_or_ratio": item["nixl_desc_reduction_x"],
                "note": f"tail_folded={item['mooncake_tail_folded']}",
            }
        )
    for item in results["deadline_arbitration"]:
        rows.append(
            {
                "section": "deadline_arbitration",
                "case_id": item["case_id"],
                "primary_metric": item["mooncake_urgent_slo_misses"],
                "secondary_metric": item["fifo_slo_misses"],
                "delta_or_ratio": item["mooncake_urgent_max_mlu"],
                "note": f"eq_misses={item['equal_split_slo_misses']}",
            }
        )
    for item in results["hicache_pd_pipeline"]:
        rows.append(
            {
                "section": "hicache_pd_pipeline",
                "case_id": item["case_id"],
                "primary_metric": item["hicache_gated_ttft_ms"],
                "secondary_metric": item["no_hicache_ttft_ms"],
                "delta_or_ratio": item["ttft_speedup_x"],
                "note": item["bottleneck"],
            }
        )
    return rows


def _format_csv(rows: list[dict[str, Any]]) -> str:
    fieldnames = ["section", "case_id", "primary_metric", "secondary_metric", "delta_or_ratio", "note"]
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true", help="Verify committed outputs match without writing")
    args = parser.parse_args(argv)

    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    results = evaluate_fixture(fixture)
    json_text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    csv_text = _format_csv(_to_csv_rows(results))

    if args.check:
        if not args.json_out.exists() or not args.csv_out.exists():
            raise SystemExit("Committed results.json or results.csv missing")
        if args.json_out.read_text(encoding="utf-8") != json_text:
            raise SystemExit("results.json does not match deterministic evaluation")
        if args.csv_out.read_text(encoding="utf-8") != csv_text:
            raise SystemExit("results.csv does not match deterministic evaluation")
        print("own-offline disaggregated_kv_routing outputs verified")
        return 0

    args.json_out.write_text(json_text, encoding="utf-8")
    args.csv_out.write_text(csv_text, encoding="utf-8")
    print("own-offline disaggregated_kv_routing outputs generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
