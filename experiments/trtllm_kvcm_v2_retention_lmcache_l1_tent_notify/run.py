#!/usr/bin/env python3
"""Deterministic offline model for Phase 34 Topic 3:
1. TensorRT-LLM KVCM V2 native retention priority/duration, cold_tier_offload_min_priority=30, and Helix CP scaling (TensorRT-LLM@41913522c7)
2. LMCache MP shared L1 MemoryOrchestrator RegionState (WRITING->VALID->CONSUMED, incarnation fencing, ReplyWindow idempotency) (LMCache@258746ce77)
3. Mooncake TENT ControlService::onNotify / ControlClient::notify structured failure reporting vs silent empty-response ack (Mooncake@5cf142e344)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def round_up_pow2(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def simulate_trtllm_kvcm_v2_retention() -> dict[str, Any]:
    cold_tier_offload_min_priority = 30
    default_priority = 35

    # Simulate 6 unreferenced GPU pages being evicted under memory pressure:
    # Pages carry priorities: [80 (system prefix), 50 (RAG context), 35 (default), 25 (ephemeral cot), 10 (scratch), 20 (rejected draft)]
    evicted_pages = [
        {"page_id": 0, "role": "system_prefix", "priority": 80, "bytes": 262144},
        {"page_id": 1, "role": "rag_context", "priority": 50, "bytes": 262144},
        {"page_id": 2, "role": "default_prompt", "priority": default_priority, "bytes": 262144},
        {"page_id": 3, "role": "ephemeral_cot", "priority": 25, "bytes": 262144},
        {"page_id": 4, "role": "scratch_chunk", "priority": 10, "bytes": 262144},
        {"page_id": 5, "role": "draft_tail", "priority": 20, "bytes": 262144},
    ]

    offloaded_bytes_without_gate = sum(p["bytes"] for p in evicted_pages)
    offloaded_pages_with_gate = [p for p in evicted_pages if p["priority"] >= cold_tier_offload_min_priority]
    dropped_pages_with_gate = [p for p in evicted_pages if p["priority"] < cold_tier_offload_min_priority]
    offloaded_bytes_with_gate = sum(p["bytes"] for p in offloaded_pages_with_gate)
    cold_write_reduction_pct = round(
        (offloaded_bytes_without_gate - offloaded_bytes_with_gate) / offloaded_bytes_without_gate * 100.0, 2
    )

    # Helix CP super-block scaling: rank-local prompt_len=512, tokens_per_block=64, helix_cp_size=4
    rank_prompt_len = 512
    tokens_per_block = 64
    helix_cp_size = 4
    prompt_blocks = (rank_prompt_len + tokens_per_block - 1) // tokens_per_block
    prompt_end = prompt_blocks * tokens_per_block
    local_range = (0, 256, 85)  # (token_start, token_end, priority)
    scaled_global_range = (
        local_range[0] * helix_cp_size,
        min(prompt_end, local_range[1]) * helix_cp_size,
        local_range[2],
    )

    return {
        "cold_tier_offload_min_priority": cold_tier_offload_min_priority,
        "default_retention_priority": default_priority,
        "total_evicted_pages": len(evicted_pages),
        "offloaded_pages_count": len(offloaded_pages_with_gate),
        "dropped_pages_count": len(dropped_pages_with_gate),
        "offloaded_bytes_without_gate": offloaded_bytes_without_gate,
        "offloaded_bytes_with_gate": offloaded_bytes_with_gate,
        "cold_tier_write_reduction_pct": cold_write_reduction_pct,
        "helix_cp_local_range": list(local_range),
        "helix_cp_scaled_global_range": list(scaled_global_range),
        "prefetch_claims_retention": False,
    }


def simulate_lmcache_memory_orchestrator() -> dict[str, Any]:
    alignment = 4096
    capacity_bytes = 16 * alignment  # 65536 B
    allocated_bytes = 0
    valid_bytes = 0
    consumed_count = 0

    # Step 1: Client "mp-worker-0" incarnation 1 reserves 3 objects:
    # obj_a (3000 B -> 4096 B), obj_b (5000 B -> 8192 B), obj_c (4096 B -> 4096 B)
    req_sizes = [("obj_a", 3000), ("obj_b", 5000), ("obj_c", 4096)]
    objects: dict[str, dict[str, Any]] = {}
    for key, payload in req_sizes:
        ext_len = round_up_pow2(payload, alignment)
        objects[key] = {
            "state": "WRITING",
            "offset": allocated_bytes,
            "length": ext_len,
            "payload_bytes": payload,
            "incarnation": 1,
        }
        allocated_bytes += ext_len

    # FinishWrite on obj_a -> VALID; obj_b and obj_c remain WRITING when incarnation 1 crashes
    objects["obj_a"]["state"] = "VALID"
    valid_bytes += objects["obj_a"]["length"]

    # Step 2: Client "mp-worker-0" restarts with incarnation 2 -> retires incarnation 1's WRITING objects to CONSUMED
    retired_writes = 0
    for key in list(objects.keys()):
        if objects[key]["incarnation"] == 1 and objects[key]["state"] == "WRITING":
            del objects[key]
            retired_writes += 1
            consumed_count += 1

    # Step 3: Batch atomic reservation check: requesting [obj_d (32768 B), obj_e (24576 B)] = 57344 B > remaining 49152 B
    # Neither is allocated (all-or-nothing batch atomic OUT_OF_SPACE)
    remaining_bytes = capacity_bytes - allocated_bytes
    oversized_batch_needed = round_up_pow2(32768, alignment) + round_up_pow2(24576, alignment)
    oversized_batch_granted = oversized_batch_needed <= remaining_bytes

    # Step 4: OrchestratorServicer _ReplyWindow (request_id -> (sha256, reply_bytes)) idempotency check
    reply_window: dict[int, tuple[str, str]] = {}
    req1_bytes = b"reserve_write:obj_a:3000"
    req1_hash = hashlib.sha256(req1_bytes).hexdigest()
    reply_window[101] = (req1_hash, "WRITE_GRANTED:offset=0")

    # Identical retry of request_id=101 -> replays cached response
    retry_identical_status = "REPLAYED_OK" if reply_window[101][0] == hashlib.sha256(req1_bytes).hexdigest() else "ALREADY_EXISTS"
    # Mutated payload reusing request_id=101 -> rejected with ALREADY_EXISTS
    mutated_bytes = b"reserve_write:obj_a:8192"
    retry_mutated_status = "REPLAYED_OK" if reply_window[101][0] == hashlib.sha256(mutated_bytes).hexdigest() else "ALREADY_EXISTS"

    return {
        "capacity_bytes": capacity_bytes,
        "alignment_bytes": alignment,
        "allocated_bytes_after_inc1": allocated_bytes,
        "valid_bytes_after_inc1": valid_bytes,
        "retired_writes_on_inc2_register": retired_writes,
        "consumed_count": consumed_count,
        "oversized_batch_needed_bytes": oversized_batch_needed,
        "remaining_bytes": remaining_bytes,
        "oversized_batch_allocated_any": oversized_batch_granted,
        "idempotent_identical_retry": retry_identical_status,
        "idempotent_mutated_retry": retry_mutated_status,
    }


def simulate_mooncake_tent_notify() -> dict[str, Any]:
    # Mooncake TENT ControlService::onNotify & ControlClient::notify (Mooncake@5cf142e344)
    cases = [
        {"case": "valid_delivered", "cb_registered": True, "json_valid": True, "cb_rc": 0},
        {"case": "startup_no_callback", "cb_registered": False, "json_valid": True, "cb_rc": 0},
        {"case": "malformed_json_payload", "cb_registered": True, "json_valid": False, "cb_rc": 0},
        {"case": "callback_nonzero_rc", "cb_registered": True, "json_valid": True, "cb_rc": -5},
    ]

    results = []
    silent_false_acks_legacy = 0
    silent_false_acks_fixed = 0

    for c in cases:
        is_actual_delivery = c["cb_registered"] and c["json_valid"] and c["cb_rc"] == 0
        # Legacy always left response="" -> ControlClient::notify returned Status::OK()
        legacy_status = "OK"
        if not is_actual_delivery and legacy_status == "OK":
            silent_false_acks_legacy += 1

        # Fixed onNotify + ControlClient::notify mapping:
        if not c["cb_registered"]:
            response = "notify callback not registered"
        elif not c["json_valid"]:
            response = "malformed notification: parse error"
        elif c["cb_rc"] != 0:
            response = f"notify callback failed: returned {c['cb_rc']}"
        else:
            response = ""

        if response == "":
            fixed_status = "OK"
            should_retry = False
        elif response.startswith("malformed notification: "):
            fixed_status = "InvalidArgument"
            should_retry = False  # Deterministic parse error fails identically on every retry
        else:
            fixed_status = "InternalError"
            should_retry = True  # Callback may be registered after startup or recover

        if not is_actual_delivery and fixed_status == "OK":
            silent_false_acks_fixed += 1

        results.append(
            {
                "case": c["case"],
                "legacy_status": legacy_status,
                "fixed_response": response,
                "fixed_status": fixed_status,
                "client_should_retry": should_retry,
            }
        )

    return {
        "silent_false_acks_legacy": silent_false_acks_legacy,
        "silent_false_acks_fixed": silent_false_acks_fixed,
        "cases": results,
    }


def build_summary() -> dict[str, Any]:
    kvcm = simulate_trtllm_kvcm_v2_retention()
    lmc = simulate_lmcache_memory_orchestrator()
    tent = simulate_mooncake_tent_notify()
    return {
        "experiment": "trtllm_kvcm_v2_retention_lmcache_l1_tent_notify",
        "phase": "Phase 34 (priority: 43)",
        "sources": [
            "NVIDIA/TensorRT-LLM@41913522c7f9f085c022421fb6183bd7592e076f",
            "LMCache/LMCache@258746ce7750b085a8cec89d371ac7eaa4adb94d",
            "kvcache-ai/Mooncake@5cf142e3447435faa9a9f6c846aea592bcfdc263",
        ],
        "trtllm_kvcm_v2_retention": kvcm,
        "lmcache_l1_orchestrator": lmc,
        "mooncake_tent_notify": tent,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        kvcm = summary["trtllm_kvcm_v2_retention"]
        assert kvcm["cold_tier_write_reduction_pct"] == 50.0
        assert kvcm["helix_cp_scaled_global_range"] == [0, 1024, 85]
        assert kvcm["prefetch_claims_retention"] is False

        lmc = summary["lmcache_l1_orchestrator"]
        assert lmc["allocated_bytes_after_inc1"] == 16384
        assert lmc["valid_bytes_after_inc1"] == 4096
        assert lmc["retired_writes_on_inc2_register"] == 2
        assert lmc["oversized_batch_allocated_any"] is False
        assert lmc["idempotent_identical_retry"] == "REPLAYED_OK"
        assert lmc["idempotent_mutated_retry"] == "ALREADY_EXISTS"

        tent = summary["mooncake_tent_notify"]
        assert tent["silent_false_acks_legacy"] == 3
        assert tent["silent_false_acks_fixed"] == 0

    print(json.dumps({"status": "ok", "experiment": summary["experiment"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
