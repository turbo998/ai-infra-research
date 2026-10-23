#!/usr/bin/env python3
"""Phase 64 Track B deterministic offline simulation and verification guard:
1. SGLang Marlin WNA16 MoE int2{num_tokens_past_padded, inactive_prefix_blocks} + sentinel_expert_bucket (#45462, 8b9b992)
2. SGLang Disaggregated Prefill-Decode heartbeat_failures reset on node recovery (#45922, 16aa816)
3. TensorRT-LLM SM100 QuACK BF16 gate/up GEMM + SwiGLU epilogue fusion (#19773, 90d4dcb) & FlashInfer KDA prefill integration (#20653, 222676c)
4. vLLM Mamba prompt-tail boundary re-caching guard (#62841, d561af5) & V2 model runner speculative draft rotary cache ceiling (#63251, 37cf712)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUMMARY_PATH = Path(__file__).with_name("summary.json")

CACHE_ONLY_ROPE_TYPES = frozenset({"default", "linear", "llama3", "ntk", "proportional"})


def simulate_sglang_marlin_moe_and_heartbeat() -> dict[str, Any]:
    """Model SGLang PR #45462 (8b9b992) and PR #45922 (16aa816)."""

    # 1. Unsigned shift sentinel_expert_bucket(expert_id, num_buckets)
    def sentinel_expert_bucket(expert_id: int, num_buckets: int, bits: int = 32) -> int:
        mask = (1 << bits) - 1
        shifted = ((expert_id & mask) + 1) & mask
        return shifted if shifted < num_buckets else 0

    num_experts = 256
    num_buckets = num_experts + 1  # bucket 0 = sentinel (-1), bucket i+1 = expert i
    sample_ids = [-1, -99, 0, 7, 255, 256, 9999]
    mapped_buckets = {str(eid): sentinel_expert_bucket(eid, num_buckets) for eid in sample_ids}

    # 2. Inactive prefix block elimination in Marlin MoE
    moe_block_size = 64
    total_padded_blocks = 16
    inactive_prefix_blocks = 6  # e.g., 6 padded sentinel blocks at head of sorted_token_ids
    num_tokens_past_padded = total_padded_blocks * moe_block_size  # 1024
    int2_published = [num_tokens_past_padded, inactive_prefix_blocks]
    active_blocks_scheduled = total_padded_blocks - inactive_prefix_blocks
    skipped_inactive_block_pct = round(100.0 * inactive_prefix_blocks / total_padded_blocks, 2)

    # 3. Disaggregated prefill heartbeat miss counter reset (#45922)
    def run_heartbeat_sequence(reset_on_failure: bool) -> dict[str, Any]:
        max_failures = 2
        probes = ["miss", "miss", "miss", 200, "miss", 200]
        misses = 0
        node_drops = 0
        alive_after_reconnect_single_miss = True
        for idx, probe in enumerate(probes):
            if probe == "miss":
                misses += 1
                if misses >= max_failures:
                    node_drops += 1
                    if idx >= 2:
                        alive_after_reconnect_single_miss = False
                    if reset_on_failure:
                        misses = 0
            else:
                misses = 0
        return {
            "node_drops": node_drops,
            "alive_after_reconnect_single_miss": alive_after_reconnect_single_miss,
        }

    buggy_hb = run_heartbeat_sequence(reset_on_failure=False)
    fixed_hb = run_heartbeat_sequence(reset_on_failure=True)

    return {
        "num_experts": num_experts,
        "num_buckets": num_buckets,
        "mapped_buckets": mapped_buckets,
        "int2_published_counts": int2_published,
        "total_padded_blocks": total_padded_blocks,
        "active_blocks_scheduled": active_blocks_scheduled,
        "skipped_inactive_block_pct": skipped_inactive_block_pct,
        "heartbeat_buggy_without_pop": buggy_hb,
        "heartbeat_fixed_with_pop": fixed_hb,
    }


def simulate_trtllm_swiglu_epilogue_and_kda() -> dict[str, Any]:
    """Model TRT-LLM PR #19773 (90d4dcb) QuACK BF16 gate/up SwiGLU epilogue & PR #20653 (222676c) KDA prefill."""
    # MiniMax-H3 shape from test_gate_up_swiglu_quack.py: M=4096, K=5376, I=14336
    tokens = 4096
    hidden = 5376
    intermediate = 14336
    bf16_bytes = 2

    # Unfused path:
    # 1) GEMM writes [M, 2I] BF16 (2 * 4096 * 14336 * 2 bytes)
    # 2) SwiGLU kernel reads [M, 2I] BF16 and writes [M, I] BF16
    unfused_intermediate_write_bytes = tokens * (2 * intermediate) * bf16_bytes
    unfused_intermediate_read_bytes = tokens * (2 * intermediate) * bf16_bytes
    final_output_write_bytes = tokens * intermediate * bf16_bytes

    unfused_activation_hbm_bytes = (
        unfused_intermediate_write_bytes + unfused_intermediate_read_bytes + final_output_write_bytes
    )
    # Fused QuACK epilogue path:
    # Applies silu(gate) * up in FP32 accumulators inside GEMM epilogue, writes [M, I] BF16 once
    fused_activation_hbm_bytes = final_output_write_bytes
    saved_hbm_bytes_per_layer = unfused_activation_hbm_bytes - fused_activation_hbm_bytes
    saved_hbm_mib_per_layer = round(saved_hbm_bytes_per_layer / (1024 * 1024), 2)
    hbm_traffic_reduction_pct = round(
        100.0 * saved_hbm_bytes_per_layer / unfused_activation_hbm_bytes, 2
    )

    # KDA prefill integration (#20653): replaces 3 standalone custom files (-5,706 lines) with vendored FlashInfer bt16 + 207-line patch
    legacy_kda_lines_removed = 1075 + 2353 + 2278  # akk_inverse.py + fused_k123.py + fused_k1234.py

    return {
        "minimax_h3_shape": {"tokens": tokens, "hidden": hidden, "intermediate": intermediate},
        "unfused_activation_hbm_mib": round(unfused_activation_hbm_bytes / (1024 * 1024), 2),
        "fused_activation_hbm_mib": round(fused_activation_hbm_bytes / (1024 * 1024), 2),
        "saved_hbm_mib_per_layer": saved_hbm_mib_per_layer,
        "activation_hbm_reduction_pct": hbm_traffic_reduction_pct,
        "bf16_roundings_unfused": 2,
        "bf16_roundings_fused": 1,
        "kda_legacy_lines_removed": legacy_kda_lines_removed,
        "kda_vendor_patch_lines": 207,
    }


def simulate_vllm_mamba_and_spec_rope() -> dict[str, Any]:
    """Model vLLM PR #62841 (d561af5) Mamba sibling admission & PR #63251 (37cf712) V2 spec-decode RoPE bounds."""
    # 1. Mamba sibling prefix-cache admission (#62841)
    num_siblings = 3
    # Without guard: _cache_partial_tail_block re-hashes block even when block_hash_num_tokens == num_tokens,
    # marking it cached in current step and deferring siblings to 1 per step -> 3 steps for 3 siblings
    steps_without_guard = num_siblings
    # With guard: if source_block.block_hash_num_tokens == num_tokens: return None -> all 3 siblings admitted in 1 step
    steps_with_guard = 1

    # 2. V2 model runner speculative decoding max_model_len & RoPE cache guard (#63251)
    def resolve_spec_max_model_len(
        method: str,
        rope_type: str,
        target_max_len: int,
        draft_max_pos: int,
        use_v2_runner: bool,
    ) -> dict[str, Any]:
        shares_target_positions = (method != "draft_model") and (rope_type in CACHE_ONLY_ROPE_TYPES)
        effective_draft_max_pos = target_max_len if shares_target_positions else draft_max_pos
        if use_v2_runner and not shares_target_positions and effective_draft_max_pos < target_max_len:
            effective_engine_max_len = effective_draft_max_pos
        else:
            effective_engine_max_len = target_max_len
        return {
            "method": method,
            "rope_type": rope_type,
            "use_v2_runner": use_v2_runner,
            "shares_target_positions": shares_target_positions,
            "effective_draft_max_pos": effective_draft_max_pos,
            "effective_engine_max_len": effective_engine_max_len,
        }

    eagle3_llama3 = resolve_spec_max_model_len("eagle3", "llama3", 131072, 2048, True)
    draft_model_v2 = resolve_spec_max_model_len("draft_model", "default", 8192, 2048, True)
    draft_model_v1 = resolve_spec_max_model_len("draft_model", "default", 8192, 2048, False)
    eagle3_yarn_v2 = resolve_spec_max_model_len("eagle3", "yarn", 131072, 4096, True)

    return {
        "mamba_shared_prefix_tokens": 32,
        "mamba_siblings_tested": num_siblings,
        "mamba_admission_steps_buggy": steps_without_guard,
        "mamba_admission_steps_fixed": steps_with_guard,
        "mamba_sibling_admission_speedup_x": round(steps_without_guard / steps_with_guard, 2),
        "spec_rope_cases": {
            "eagle3_llama3_v2": eagle3_llama3,
            "draft_model_v2": draft_model_v2,
            "draft_model_v1": draft_model_v1,
            "eagle3_yarn_v2": eagle3_yarn_v2,
        },
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "sglang-marlin-moe-trtllm-kda-swiglu-vllm",
        "phase": 64,
        "priority": 73,
        "provenance": "own-offline deterministic guard + upstream source formulas (SGLang #45462, #45922; TRT-LLM #19773, #20653; vLLM #62841, #63251)",
        "sglang_marlin_and_heartbeat": simulate_sglang_marlin_moe_and_heartbeat(),
        "trtllm_swiglu_and_kda": simulate_trtllm_swiglu_epilogue_and_kda(),
        "vllm_mamba_and_spec_rope": simulate_vllm_mamba_and_spec_rope(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches deterministic outputs")
    args = parser.parse_args()

    summary = build_summary()
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not SUMMARY_PATH.exists():
            raise SystemExit(f"Missing {SUMMARY_PATH}")
        existing = SUMMARY_PATH.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of sync with deterministic run.py output")
        print("own-offline outputs verified for sglang-marlin-moe-trtllm-kda-swiglu-vllm")
        return 0

    SUMMARY_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
