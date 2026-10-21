"""Deterministic model for Megatron-LM Hybrid DSA-GQA + GDN CUDA-Graph Prefill, Mooncake Store KV Reshard Planner + TenantQuotaPolicy, and NIXL Telemetry Execution Counter."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


def build_gdp_chunk_descriptors(
    cu_seqlens: list[int],
    chunk_size: int = 64,
    padded_chunk_capacity: int = 16,
) -> dict[str, object]:
    """Model Megatron-LM PR #8353 GDN prefill CUDA-graph-safe fixed-capacity GDP chunk descriptors."""
    num_seqs = len(cu_seqlens) - 1
    active_chunks: list[tuple[int, int, int]] = []
    chunk_offsets: list[int] = [0]
    for seq_idx in range(num_seqs):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        n_chunks = math.ceil(seq_len / chunk_size) if seq_len > 0 else 0
        for local_chunk in range(n_chunks):
            active_chunks.append((seq_idx, local_chunk, cu_seqlens[seq_idx] + local_chunk * chunk_size))
        chunk_offsets.append(len(active_chunks))

    if len(active_chunks) > padded_chunk_capacity:
        raise ValueError("Active chunks exceed CUDA graph static descriptor capacity")

    # Pad up to fixed capacity so workspace and launch grid stay invariant across replays
    padded_indices = [
        (seq_idx, local_chunk) for seq_idx, local_chunk, _ in active_chunks
    ] + [(-1, -1)] * (padded_chunk_capacity - len(active_chunks))

    return {
        "cu_seqlens": cu_seqlens,
        "chunk_size": chunk_size,
        "num_householder": 1,
        "active_chunk_count": len(active_chunks),
        "padded_chunk_capacity": padded_chunk_capacity,
        "chunk_offsets": chunk_offsets,
        "first_active_descriptors": padded_indices[: len(active_chunks)],
        "padding_slots": padded_chunk_capacity - len(active_chunks),
    }


def simulate_dsa_gqa_indexer_loss_logging(
    raw_losses: list[float],
    loss_scale: float,
) -> dict[str, object]:
    """Model Megatron-LM PR #7156 `DSGroupedSelfAttention` + `DSAIndexerLossLoggingHelper` raw_values tracking."""
    scaled_losses = [val * loss_scale for val in raw_losses]
    buggy_logged_mean = sum(scaled_losses) / len(scaled_losses)
    fixed_raw_mean = sum(raw_losses) / len(raw_losses)
    return {
        "num_layers_logged": len(raw_losses),
        "loss_scale": loss_scale,
        "scaled_backward_sum": round(sum(scaled_losses), 6),
        "buggy_scaled_logged_mean": round(buggy_logged_mean, 6),
        "fixed_raw_values_mean": round(fixed_raw_mean, 6),
        "scale_distortion_factor": round(buggy_logged_mean / fixed_raw_mean, 4),
    }


@dataclass(frozen=True)
class PlacementPart:
    participant_id: str
    dp: int
    tp: int
    pp: int
    replica_ordinal: int
    layer_ids: tuple[int, ...]
    head_start: int
    head_count: int


def plan_mooncake_store_upload_and_load(
    source_parts: list[PlacementPart],
    target_parts: list[PlacementPart],
    *,
    token_count: int = 256,
    head_dim: int = 128,
    itemsize: int = 2,
) -> dict[str, object]:
    """Model Mooncake PR #4762 Store KV reuse upload/load planner (`replica_ordinal == 0` upload dedup + interval cover)."""
    # Upload selects replica_ordinal == 0 only, sorted by (layer_ids, head_start)
    upload_parts = sorted(
        [p for p in source_parts if p.replica_ordinal == 0 and p.layer_ids],
        key=lambda p: (p.layer_ids, p.head_start),
    )
    head_stride_bytes = token_count * head_dim * itemsize

    # Build store fragments per (layer, component)
    fragments_by_layer_comp: dict[tuple[int, str], list[dict[str, object]]] = {}
    for shard_idx, part in enumerate(upload_parts):
        part_offset_cursor = 0
        for layer in part.layer_ids:
            for comp in ("KEY", "VALUE"):
                frag = {
                    "shard_id": f"{part.tp}_{part.pp}",
                    "layer": layer,
                    "component": comp,
                    "head_start": part.head_start,
                    "head_stop": part.head_start + part.head_count,
                    "part_offset": part_offset_cursor,
                    "head_stride_bytes": head_stride_bytes,
                }
                fragments_by_layer_comp.setdefault((layer, comp), []).append(frag)
                part_offset_cursor += part.head_count * head_stride_bytes

    # Load fills EVERY target replica (replicas=True) by intersecting head intervals
    load_ranges: list[dict[str, object]] = []
    useful_bytes = 0
    for tpart in target_parts:
        t_start = tpart.head_start
        t_stop = tpart.head_start + tpart.head_count
        for layer in tpart.layer_ids:
            for comp in ("KEY", "VALUE"):
                for frag in fragments_by_layer_comp[(layer, comp)]:
                    f_start = int(frag["head_start"])
                    f_stop = int(frag["head_stop"])
                    ov_start = max(t_start, f_start)
                    ov_stop = min(t_stop, f_stop)
                    if ov_start < ov_stop:
                        rel_offset = int(frag["part_offset"]) + (ov_start - f_start) * head_stride_bytes
                        nbytes = (ov_stop - ov_start) * head_stride_bytes
                        useful_bytes += nbytes
                        load_ranges.append(
                            {
                                "target_participant": tpart.participant_id,
                                "replica_ordinal": tpart.replica_ordinal,
                                "source_shard": frag["shard_id"],
                                "layer": layer,
                                "component": comp,
                                "head_start": ov_start,
                                "head_count": ov_stop - ov_start,
                                "part_offset": rel_offset,
                                "bytes": nbytes,
                            }
                        )

    return {
        "source_total_parts": len(source_parts),
        "upload_writer_count": len(upload_parts),
        "upload_writers": [p.participant_id for p in upload_parts],
        "target_total_parts": len(target_parts),
        "load_range_count": len(load_ranges),
        "useful_bytes": useful_bytes,
    }


def simulate_mooncake_tenant_quota_lifecycle(
    quota_bytes: int,
    initial_write_bytes: int,
    upsert_reserved_bytes: int,
    upsert_actual_bytes: int,
    *,
    watermark_ratio: float = 0.85,
    eviction_ratio: float = 0.15,
) -> dict[str, object]:
    """Model Mooncake PR #4809 `TenantQuotaPolicy` (`Held`, `Carried`, `OnWriteRelease`, `OnMaintenance`)."""
    # Step 1: initial object write settles
    held = initial_write_bytes
    carried = 0
    charged = held

    # Step 2: upsert replaces object while replacement replica is still PROCESSING
    # OnReplace adds old object's charge into Carried; OnReserve adds new reservation into Held
    carried += initial_write_bytes
    held += upsert_reserved_bytes
    charged += upsert_reserved_bytes
    # First OnWriteRelease while processing memory replica is still active:
    # owed = MemoryCharge(processing_new) + carried
    owed_while_processing = upsert_actual_bytes + carried
    if owed_while_processing > held:
        charged += owed_while_processing - held
    elif owed_while_processing < held:
        charged -= held - owed_while_processing
    held = owed_while_processing
    charged_during_processing_upsert = charged

    # Step 3: processing replica finishes -> HasProcessingMemoryReplica is False -> carried = 0
    carried = 0
    owed_after_settle = upsert_actual_bytes
    charged -= held - owed_after_settle
    held = owed_after_settle

    # Step 4: Maintenance watermark eviction check
    used_ratio = charged / quota_bytes
    target_ratio = max(0.0, watermark_ratio - eviction_ratio)
    evict_target_bytes = (
        int((used_ratio - target_ratio) * quota_bytes)
        if used_ratio > watermark_ratio
        else 0
    )

    return {
        "quota_bytes": quota_bytes,
        "charged_after_initial": initial_write_bytes,
        "charged_during_processing_upsert": charged_during_processing_upsert,
        "charged_after_upsert_settle": charged,
        "held_word0": held,
        "carried_word1": carried,
        "used_ratio": round(used_ratio, 4),
        "watermark_ratio": watermark_ratio,
        "target_ratio": round(target_ratio, 4),
        "evict_target_bytes": evict_target_bytes,
    }


def simulate_nixl_xfer_telemetry_ids(
    drawn_random_ids: list[int],
    submission_outcomes: list[str],
) -> dict[str, object]:
    """Model NIXL PR #2479 `nixl.xfer_request_id` (redraw zero) and `nixl.xfer_execution` counter."""
    idx = 0
    redraws = 0
    while True:
        candidate = drawn_random_ids[idx]
        idx += 1
        if candidate != 0:
            request_id = candidate
            break
        redraws += 1

    execution_counter = 0
    completed_events: list[dict[str, int]] = []
    for outcome in submission_outcomes:
        if outcome == "validation_failed_before_postXfer":
            continue
        execution_counter += 1
        completed_events.append(
            {
                "xfer_request_id": request_id,
                "xfer_execution": execution_counter,
            }
        )

    return {
        "xfer_request_id": request_id,
        "zero_redraws": redraws,
        "total_post_attempts": len(submission_outcomes),
        "final_xfer_execution": execution_counter,
        "completed_events": completed_events,
    }


def build_summary() -> dict[str, object]:
    gdp_graph = build_gdp_chunk_descriptors(
        cu_seqlens=[0, 110, 174, 310],
        chunk_size=64,
        padded_chunk_capacity=12,
    )
    dsa_loss = simulate_dsa_gqa_indexer_loss_logging(
        raw_losses=[0.42, 0.38, 0.35, 0.45],
        loss_scale=0.1,
    )

    # Source: TP=2, PP=1, with 2 KV head replicas per shard (4 total parts, only 2 have replica_ordinal==0)
    source_parts = [
        PlacementPart("src_tp0_rep0", dp=0, tp=0, pp=0, replica_ordinal=0, layer_ids=(0, 1), head_start=0, head_count=4),
        PlacementPart("src_tp0_rep1", dp=0, tp=0, pp=0, replica_ordinal=1, layer_ids=(0, 1), head_start=0, head_count=4),
        PlacementPart("src_tp1_rep0", dp=0, tp=1, pp=0, replica_ordinal=0, layer_ids=(0, 1), head_start=4, head_count=4),
        PlacementPart("src_tp1_rep1", dp=0, tp=1, pp=0, replica_ordinal=1, layer_ids=(0, 1), head_start=4, head_count=4),
    ]
    # Target: TP=4, PP=1, with 2 replicas for each 2-head slice (8 target parts)
    target_parts: list[PlacementPart] = []
    for tp in range(4):
        for rep in (0, 1):
            target_parts.append(
                PlacementPart(
                    f"dst_tp{tp}_rep{rep}",
                    dp=0,
                    tp=tp,
                    pp=0,
                    replica_ordinal=rep,
                    layer_ids=(0, 1),
                    head_start=tp * 2,
                    head_count=2,
                )
            )
    mooncake_reshard = plan_mooncake_store_upload_and_load(source_parts, target_parts)
    mooncake_quota = simulate_mooncake_tenant_quota_lifecycle(
        quota_bytes=1000,
        initial_write_bytes=400,
        upsert_reserved_bytes=850,
        upsert_actual_bytes=900,
        watermark_ratio=0.85,
        eviction_ratio=0.15,
    )
    nixl_telemetry = simulate_nixl_xfer_telemetry_ids(
        drawn_random_ids=[0, 0, 987654321012345],
        submission_outcomes=[
            "validation_failed_before_postXfer",
            "posted_initial",
            "validation_failed_before_postXfer",
            "reposted_step_2",
        ],
    )

    return {
        "source_commits": [
            "NVIDIA/Megatron-LM@d14b699755061d55cb918f68500fba24401a1846",
            "NVIDIA/Megatron-LM@29e1a7b6697c61d184c95726865d417fc51b8574",
            "kvcache-ai/Mooncake@e2d6e8e0f2df113d84f9e39379f98c4b67b199ab",
            "kvcache-ai/Mooncake@44c4d9cf4ace3ce2955312a2467b1305d910dd48",
            "ai-dynamo/nixl@03d9b74648cc06daef8aa71f15a53c5a98be1319",
        ],
        "megatron_hybrid_dsa_gqa_and_gdn": {
            "gdp_prefill_cudagraph": gdp_graph,
            "dsa_indexer_loss_logging": dsa_loss,
        },
        "mooncake_store_reshard_and_quota": {
            "reshard_plan": mooncake_reshard,
            "tenant_quota_lifecycle": mooncake_quota,
        },
        "nixl_xfer_telemetry": nixl_telemetry,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        gdp = summary["megatron_hybrid_dsa_gqa_and_gdn"]["gdp_prefill_cudagraph"]
        assert gdp["num_householder"] == 1
        assert gdp["active_chunk_count"] == 6 and gdp["padding_slots"] == 6
        loss = summary["megatron_hybrid_dsa_gqa_and_gdn"]["dsa_indexer_loss_logging"]
        assert loss["fixed_raw_values_mean"] == 0.4 and loss["buggy_scaled_logged_mean"] == 0.04
        reshard = summary["mooncake_store_reshard_and_quota"]["reshard_plan"]
        assert reshard["upload_writer_count"] == 2
        assert reshard["load_range_count"] == 32  # 8 target parts * 2 layers * 2 components
        quota = summary["mooncake_store_reshard_and_quota"]["tenant_quota_lifecycle"]
        assert quota["charged_during_processing_upsert"] == 1300
        assert quota["charged_after_upsert_settle"] == 900
        assert quota["evict_target_bytes"] == 200
        nixl = summary["nixl_xfer_telemetry"]
        assert nixl["zero_redraws"] == 2 and nixl["final_xfer_execution"] == 2

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
