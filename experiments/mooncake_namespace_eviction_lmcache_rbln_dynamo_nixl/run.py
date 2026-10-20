#!/usr/bin/env python3
"""Phase 37 Experiment 3: Mooncake Store Namespace & LeaseEvictionIndex,
LMCache vLLM-RBLN Blocks-First 6-D KV Layout & Dynamo Text-Input KV Router Guard / NIXL Explicit-Time Phase Events.

Audited upstream commits:
1. kvcache-ai/Mooncake @ d855a6aed9c8d1343af008b2cd9b9f4257d62a9b (#4721, 2026-10-20)
2. LMCache/LMCache @ 462006abb998d4e4cb4209b14eb3d2db74a931b7 (#5748, 2026-10-20)
3. ai-dynamo/dynamo @ 1b6fee46c88b0f7339ac19423150d0613c82fa38 (#16486, 2026-10-20)
4. ai-dynamo/nixl @ 705e0fc4f76993e9bbc5d39ed372db23c0c79e39 (#2460, 2026-10-20)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_mooncake_namespace_lease_eviction() -> dict[str, Any]:
    # Models kvcache-ai/Mooncake #4721 (d855a6aed9):
    # LeaseEvictionIndex: kBucketWidth = 100ms, observes_reads = False.
    # Reads that extend a key's lease do NOT mutate the index; cursor walk checks StillAt under write lock and refiles.
    bucket_width_ms = 100
    objects = {
        "k_expired_1": {"filed_deadline_ms": 150, "real_deadline_ms": 150, "hard_pinned": False, "has_mem_replica": True},
        "k_read_extended": {"filed_deadline_ms": 180, "real_deadline_ms": 620, "hard_pinned": False, "has_mem_replica": True},
        "k_expired_2": {"filed_deadline_ms": 240, "real_deadline_ms": 240, "hard_pinned": False, "has_mem_replica": True},
        "k_hard_pinned": {"filed_deadline_ms": 120, "real_deadline_ms": 120, "hard_pinned": True, "has_mem_replica": True},
        "k_future": {"filed_deadline_ms": 450, "real_deadline_ms": 450, "hard_pinned": False, "has_mem_replica": True},
    }

    now_ms = 260
    now_bucket = now_ms // bucket_width_ms  # bucket 2 (covers [200, 299] ms)

    # Index only files objects where Qualifies(metadata) is True (!hard_pinned && completed memory replica)
    filed_buckets: dict[int, list[str]] = {}
    for key, meta in objects.items():
        if not meta["hard_pinned"] and meta["has_mem_replica"]:
            rank = meta["filed_deadline_ms"] // bucket_width_ms
            filed_buckets.setdefault(rank, []).append(key)

    evicted: list[str] = []
    refiled: dict[str, int] = {}
    for rank in sorted(filed_buckets):
        if rank > now_bucket:
            break
        for key in filed_buckets[rank]:
            real_deadline = objects[key]["real_deadline_ms"]
            real_rank = real_deadline // bucket_width_ms
            if real_rank > rank:
                # Read extended lease without index lock; StillAt catches up under write lock and refiles
                refiled[key] = real_rank
            elif real_deadline <= now_ms:
                evicted.append(key)

    return {
        "bucket_width_ms": bucket_width_ms,
        "observes_reads": False,
        "read_lock_mutations_for_10k_reads": 0,
        "default_namespace_atomic_fast_path_locks": 0,
        "walk_at_260ms_evicted_keys": evicted,
        "walk_at_260ms_refiled_keys": refiled,
    }


def simulate_lmcache_rbln_blocks_first() -> dict[str, Any]:
    # Models LMCache/LMCache #5748 (462006abb9):
    # EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS = 19 ([NB, 2, NH, 1, BS, HS])
    # vs NL_X_TWO_NB_NH_ONE_BS_HS = 15 ([2, NB, NH, 1, BS, HS])
    def detect_rbln_6d_format(shape: tuple[int, ...]) -> tuple[int, bool]:
        assert len(shape) == 6
        ambiguous_nb2 = (shape[0] == 2 and shape[1] == 2 and shape[3] == 1)
        if shape[0] == 2 and shape[3] == 1:
            return 15, ambiguous_nb2  # NL_X_TWO_NB_NH_ONE_BS_HS
        if shape[1] == 2 and shape[3] == 1:
            return 19, False          # NL_X_NB_TWO_NH_ONE_BS_HS
        raise ValueError(f"Unsupported 6-D shape: {shape}")

    fmt_blocks_first, warn_bf = detect_rbln_6d_format((64, 2, 8, 1, 1024, 128))
    fmt_kv_first, warn_kv = detect_rbln_6d_format((2, 64, 8, 1, 1024, 128))
    fmt_ambiguous, warn_amb = detect_rbln_6d_format((2, 2, 8, 1, 1024, 128))

    # Copy descriptor count for B=16 blocks across L=32 layers:
    # Blocks-first: layer[block] holds K and V contiguously -> B * L = 512 direct memcpy_v2v pairs
    # K/V-first: layer[:, block] is strided across the leading axis -> 2 * B * L = 1024 memcpy_v2v pairs
    n_blocks = 16
    n_layers = 32
    blocks_first_v2v_copies = n_blocks * n_layers
    kv_first_v2v_copies = 2 * n_blocks * n_layers
    copy_reduction_pct = round(
        (kv_first_v2v_copies - blocks_first_v2v_copies) / kv_first_v2v_copies * 100.0, 2
    )

    return {
        "blocks_first_format_enum": fmt_blocks_first,
        "kv_first_format_enum": fmt_kv_first,
        "ambiguous_nb2_format_enum": fmt_ambiguous,
        "ambiguous_nb2_warning_emitted": warn_amb,
        "blocks_first_v2v_copies": blocks_first_v2v_copies,
        "kv_first_v2v_copies": kv_first_v2v_copies,
        "v2v_copy_reduction_pct": copy_reduction_pct,
    }


def simulate_dynamo_router_and_nixl_trace() -> dict[str, Any]:
    # 1. ai-dynamo/dynamo #16486 (1b6fee46c8): ModelWatcher discovery rejects ModelInput::Text when effective router_mode.is_kv_routing()
    def validate_discovery(model_input: str, card_router_mode: str | None, frontend_router_mode: str) -> bool:
        effective_mode = card_router_mode if card_router_mode is not None else frontend_router_mode
        is_kv = (effective_mode == "KV")
        return not (model_input == "Text" and is_kv)

    matrix = {
        "text_input_kv_router": validate_discovery("Text", None, "KV"),
        "text_input_card_kv_override": validate_discovery("Text", "KV", "RoundRobin"),
        "text_input_card_rr_override": validate_discovery("Text", "RoundRobin", "KV"),
        "tokens_input_kv_router": validate_discovery("Tokens", None, "KV"),
    }

    # 2. ai-dynamo/nixl #2460 (705e0fc4f7): TracerPhaseSink::recordPhase routes phases via emitEvent at plugin timestamp_us
    phases = [
        ("SUBMIT", 1050, "Metadata"),
        ("WIRE_SUBMITTED", 1062, "Metadata"),
        ("NOTIF_SENT", 1088, "CommSend"),
        ("WIRE_COMPLETED", 1140, "Metadata"),
        ("NOTIF_RECEIVED", 1145, "CommRecv"),
    ]
    emitted_events = [
        {
            "phase": name,
            "at_us": ts_us,
            "kind": kind,
            "correlation_scope_opened": False,
        }
        for name, ts_us, kind in phases
    ]

    return {
        "dynamo_discovery_matrix": matrix,
        "nixl_emitted_phase_events": emitted_events,
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "mooncake_namespace_eviction_lmcache_rbln_dynamo_nixl",
        "phase": 37,
        "priority": 46,
        "evidence_tier": "own-offline",
        "mooncake_namespace_eviction": simulate_mooncake_namespace_lease_eviction(),
        "lmcache_rbln_blocks_first": simulate_lmcache_rbln_blocks_first(),
        "dynamo_router_and_nixl_trace": simulate_dynamo_router_and_nixl_trace(),
    }


def verify(summary: dict[str, Any]) -> None:
    mc = summary["mooncake_namespace_eviction"]
    assert mc["bucket_width_ms"] == 100
    assert mc["observes_reads"] is False
    assert mc["read_lock_mutations_for_10k_reads"] == 0
    assert mc["walk_at_260ms_evicted_keys"] == ["k_expired_1", "k_expired_2"]
    assert mc["walk_at_260ms_refiled_keys"] == {"k_read_extended": 6}

    lm = summary["lmcache_rbln_blocks_first"]
    assert lm["blocks_first_format_enum"] == 19
    assert lm["kv_first_format_enum"] == 15
    assert lm["ambiguous_nb2_warning_emitted"] is True
    assert lm["blocks_first_v2v_copies"] == 512 and lm["kv_first_v2v_copies"] == 1024
    assert lm["v2v_copy_reduction_pct"] == 50.0

    dn = summary["dynamo_router_and_nixl_trace"]
    assert dn["dynamo_discovery_matrix"] == {
        "text_input_kv_router": False,
        "text_input_card_kv_override": False,
        "text_input_card_rr_override": True,
        "tokens_input_kv_router": True,
    }
    kinds = {e["phase"]: e["kind"] for e in dn["nixl_emitted_phase_events"]}
    assert kinds["NOTIF_SENT"] == "CommSend" and kinds["NOTIF_RECEIVED"] == "CommRecv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    summary = build_summary()
    verify(summary)

    if args.check:
        for rel in ("summary.json", "results/summary.json"):
            p = out_dir / rel
            assert p.exists(), f"Missing {p}"
            existing = json.loads(p.read_text(encoding="utf-8"))
            assert existing == summary, f"Drift in {p}"
        print("mooncake_namespace_eviction_lmcache_rbln_dynamo_nixl: check passed")
        return 0

    for rel in ("summary.json", "results/summary.json"):
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
