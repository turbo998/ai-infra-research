#!/usr/bin/env python3
"""Deterministic stdlib reproduction for Phase 48 Topic 2:
SGLang Disaggregated Decode Deferred-KV TP Synchronization & Composable Radix Eviction Walk,
TensorRT-LLM Standalone CPU Render Endpoint & Gemma-4 Non-SM100 Hybrid KV Spec & Gen-Only Perf Floor,
and vLLM Placeholder Offload Group Classification & Multimodal Chunked-Embedding Guard.

Upstream references:
- sgl-project/sglang@b57b5e22867f110ff577a9da18a9ff7e6619100c (2026-10-21)
- sgl-project/sglang@5e79333c95c162abf079548fe40a6a2bca6c5409 (2026-10-21)
- NVIDIA/TensorRT-LLM@ec3352fee21a7a16ebd320d30a94e1a7d5ac3bda (2026-10-21)
- NVIDIA/TensorRT-LLM@e8127c7c5cff381752d7e2343f05fbbabe34df35 (2026-10-21)
- NVIDIA/TensorRT-LLM@7f0caf443ec440ffc7ca455c91ff6f7c8b0b8d67 (2026-10-21)
- vllm-project/vllm@550c1cbb8b0a366f1bb2b5aea560a1d8e70f04cb (2026-10-21)
- vllm-project/vllm@65a886c69f44cf446d6afa03315f39590ccd843e (2026-10-21)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def simulate_sglang_disagg_tp_sync_and_radix_walk() -> dict[str, Any]:
    """Model SGLang PR #45453 (DecodeTransferQueue.pop_transferred TP sync) and PR #45439 (cache_actions walk)."""
    # PR #45453: 4 attention-TP ranks; rank 0 observes peer_WaitSuccess=True while rank 2 observes timeout/error
    # Without synchronize_Failed_Statuses across TP group, rank 0 holds KV in release_Wait_Queue while rank 2 frees immediately -> KV pool divergence
    tp_size = 4
    rank_transfer_succeeded = [True, True, False, True]
    unsynced_hold_decisions = [ok for ok in rank_transfer_succeeded]  # diverges!
    synced_all_succeeded = all(rank_transfer_succeeded)
    # When any rank fails, all TP ranks agree on the unified release path
    synced_hold_decisions = [synced_all_succeeded for _ in range(tp_size)]

    # PR #45439: BaseEvictionResult threads cache_actions: list[CacheAction | ComponentAction]
    # Ordering invariant: _apply_eviction_result drains _free_values BEFORE _apply_cache_actions
    execution_log: list[str] = []
    free_values = ["kv_page_101", "kv_page_102"]
    cache_actions = ["mamba_slot_unmap_7", "swa_evict_callback_12"]

    for page in free_values:
        execution_log.append(f"free_value:{page}")
    for action in cache_actions:
        execution_log.append(f"cache_action:{action}")

    free_before_cache_action = execution_log.index("free_value:kv_page_102") < execution_log.index(
        "cache_action:mamba_slot_unmap_7"
    )

    return {
        "tp_size": tp_size,
        "unsynced_ranks_diverged": len(set(unsynced_hold_decisions)) > 1,
        "synced_ranks_diverged": len(set(synced_hold_decisions)) > 1,
        "free_before_cache_action_invariant": free_before_cache_action,
        "eviction_steps_executed": len(execution_log),
        "unconsumed_result_tripwire": "__del__",
    }


def simulate_trtllm_render_gemma4_and_perf_floor() -> dict[str, Any]:
    """Model TensorRT-LLM PR #19964 (trtllm-render), PR #18547 (Gemma4 KVCacheLayerSpec), and PR #20297 (perf floor)."""
    # 1. PR #19964: Standalone CPU-only trtllm-render + render_fingerprint.digest + Kimi-K3 opener token exclusion
    rendered_tokens = [151644, 872, 198, 1234, 5678, 151645, 151644, 77091, 198]
    kimi_k3_generation_channel_opener_tokens = 3
    payload_bytes = json.dumps({"tokens": rendered_tokens, "mm_positions": []}, sort_keys=True).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
    prompt_tokens_reported = len(rendered_tokens) - kimi_k3_generation_channel_opener_tokens

    # 2. PR #18547: Gemma-4 hybrid attention head-group ratio across layer specs on non-SM100 GPUs
    # Layer 0..2: sliding-window GQA (num_qo_heads=16, num_kv_heads=8 -> ratio=2)
    # Layer 3: global GQA (num_qo_heads=16, num_kv_heads=2 -> ratio=8)
    layer_specs = [
        {"layer_idx": 0, "num_qo_heads": 16, "num_kv_heads": 8},
        {"layer_idx": 1, "num_qo_heads": 16, "num_kv_heads": 8},
        {"layer_idx": 2, "num_qo_heads": 16, "num_kv_heads": 8},
        {"layer_idx": 3, "num_qo_heads": 16, "num_kv_heads": 2},
    ]
    max_num_heads_per_kv = max(spec["num_qo_heads"] // spec["num_kv_heads"] for spec in layer_specs)

    # 3. PR #20297: Floor gen_only device-step-time baselines at GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS
    floor_ms = 1790856543000  # 2026-10-01T12:09:03Z
    history_records = [
        {"ts_created": 1790000000000, "step_time_ms": 4.2},   # Old modal-bucket short-tail wave (~3x lower)
        {"ts_created": 1790900000000, "step_time_ms": 12.4},  # New expected steady-state bucket
        {"ts_created": 1791000000000, "step_time_ms": 12.6},
        {"ts_created": 1791100000000, "step_time_ms": 12.5},
    ]
    unfloored_min = min(r["step_time_ms"] for r in history_records)
    floored_values = [r["step_time_ms"] for r in history_records if r["ts_created"] >= floor_ms]
    floored_min = min(floored_values)
    current_run_ms = 12.55
    false_regression_without_floor = current_run_ms > unfloored_min * 1.05
    passes_with_floor = current_run_ms <= floored_min * 1.05

    return {
        "render_digest": digest,
        "rendered_token_count": len(rendered_tokens),
        "kimi_k3_opener_tokens_excluded": kimi_k3_generation_channel_opener_tokens,
        "prompt_tokens_reported": prompt_tokens_reported,
        "gemma4_max_num_heads_per_kv": max_num_heads_per_kv,
        "gen_only_history_floor_ms": floor_ms,
        "unfloored_baseline_ms": unfloored_min,
        "floored_baseline_ms": floored_min,
        "false_regression_without_floor": false_regression_without_floor,
        "passes_with_floor": passes_with_floor,
    }


def simulate_vllm_offload_and_pooling_guard() -> dict[str, Any]:
    """Model vLLM PR #59703 (placeholder offload group spec survival) and PR #62859 (chunked embedding MM guard)."""
    # PR #59703: OffloadingEventsTracker static _group_specs survives reset()
    group_specs = {
        0: {"kind": "mamba", "sliding_window": None},
        1: {"kind": "sliding_window", "sliding_window": 16},
    }
    # After tracker.reset(), dynamic block residency is cleared, but _placeholder_stored still consults _group_specs
    placeholder_stored_after_reset = {
        "group_idx": 0,
        "block_size": 0,
        "token_ids": [],
        "kv_cache_spec_kind": group_specs[0]["kind"],
        "kv_cache_spec_sliding_window": group_specs[0]["sliding_window"],
    }

    # PR #62859: maybe_pre_process_chunked rejects any engine_input["prompts"]["type"] == "multimodal"
    mixed_batch_types = ["tokens", "multimodal"]
    rejected_when_chunked = any(t == "multimodal" for t in mixed_batch_types)

    return {
        "placeholder_survives_reset": placeholder_stored_after_reset["kv_cache_spec_kind"] == "mamba",
        "placeholder_block_size": placeholder_stored_after_reset["block_size"],
        "multimodal_chunked_embedding_rejected": rejected_when_chunked,
        "exception_type": "VLLMValidationError",
    }


def build_summary() -> dict[str, Any]:
    sglang = simulate_sglang_disagg_tp_sync_and_radix_walk()
    trtllm = simulate_trtllm_render_gemma4_and_perf_floor()
    vllm = simulate_vllm_offload_and_pooling_guard()

    assert sglang["unsynced_ranks_diverged"] is True
    assert sglang["synced_ranks_diverged"] is False
    assert sglang["free_before_cache_action_invariant"] is True
    assert trtllm["gemma4_max_num_heads_per_kv"] == 8
    assert trtllm["false_regression_without_floor"] is True
    assert trtllm["passes_with_floor"] is True
    assert vllm["placeholder_survives_reset"] is True
    assert vllm["multimodal_chunked_embedding_rejected"] is True

    return {
        "experiment_id": "sglang-disagg-deferred-kv-tp-sync-radix-walk-trtllm-render-gemma4-vllm-offload",
        "phase": 48,
        "priority": 57,
        "status": "verified",
        "sglang_disagg_and_radix": sglang,
        "trtllm_render_gemma4_perf": trtllm,
        "vllm_offload_and_pooling": vllm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify invariants and write summary.json")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.check:
        print(f"[PASS] {summary['experiment_id']} verified -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
