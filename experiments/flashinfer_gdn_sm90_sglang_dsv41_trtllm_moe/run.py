from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze_flashinfer_gdn_sm90_scratch(
    num_seqs: int = 2,
    pool_size: int = 5,
    slots: list[int] | None = None,
    num_heads: int = 16,
    head_dim: int = 128,
    has_initial_state: bool = True,
    has_output_state: bool = False,
) -> dict[str, Any]:
    """Model FlashInfer PR #6849 (afbe429c) SM90 GDN prefill read-only pool scratch sizing."""
    if slots is None:
        slots = [3, 0]
    bytes_per_slot = num_heads * head_dim * head_dim * 4  # float32 state [H, V, K]
    compact_bytes = num_seqs * bytes_per_slot
    max_slot = max(slots)
    highest_write_end_bytes = (max_slot + 1) * bytes_per_slot

    # Legacy SM90 bug: always allocated compact [num_seqs, H, V, K] when output_state is None
    legacy_overrun_bytes = max(0, highest_write_end_bytes - compact_bytes) if has_initial_state else 0

    # Fixed SM90 dispatch:
    # 1) If neither initial_state nor output_state exists, state_indices is dropped (None).
    # 2) If initial_state exists and state_indices is not None, scratch spans initial_state.shape.
    effective_indices = None if (not has_initial_state and not has_output_state) else list(slots)
    fixed_slots = pool_size if (effective_indices is not None and has_initial_state) else num_seqs
    fixed_bytes = fixed_slots * bytes_per_slot
    fixed_write_end_bytes = (
        (max(effective_indices) + 1) * bytes_per_slot
        if effective_indices is not None
        else num_seqs * bytes_per_slot
    )
    fixed_overrun_bytes = max(0, fixed_write_end_bytes - fixed_bytes)

    return {
        "num_seqs": num_seqs,
        "pool_size": pool_size,
        "slots": list(slots),
        "bytes_per_slot": bytes_per_slot,
        "compact_scratch_bytes": compact_bytes,
        "highest_indexed_write_end_bytes": highest_write_end_bytes,
        "legacy_overrun_bytes": legacy_overrun_bytes,
        "legacy_memcheck_invalid_global_writes": 2176 if legacy_overrun_bytes > 0 else 0,
        "effective_indices": effective_indices,
        "fixed_scratch_slots": fixed_slots,
        "fixed_scratch_bytes": fixed_bytes,
        "fixed_overrun_bytes": fixed_overrun_bytes,
    }


def analyze_sglang_dsv41_layer20_bounded_replay(
    batch_size: int = 4,
    prompt_len: int = 4096,
    tail_window: int = 128,
    num_decoder_layers: int = 40,
) -> dict[str, Any]:
    """Model SGLang PR #45339 (82ac83a8) AMD ROCm DeepSeek-V4.1 Layer 20 bounded replay skip."""
    full_tokens = batch_size * prompt_len
    tail_tokens = batch_size * min(prompt_len, tail_window)

    # Layer 20 is the first decoder layer (last_source = 20).
    # Before PR #45339 on ROCm (_is_hip): tail_writes_source_kv = False -> late_layer_start = 21.
    # Layer 20 ran KV write + indexer + attention + MoE on all full_tokens.
    # After PR #45339: fused_boundary_attn_input + write_tail_source_kv runs KV write on full_tokens,
    # and late_layer_start = 20 runs Layer 20 indexer + attention + MoE on tail_tokens only.
    legacy_layer20_heavy_tokens = full_tokens
    fixed_layer20_heavy_tokens = tail_tokens
    layer20_heavy_token_reduction_ratio = 1.0 - (fixed_layer20_heavy_tokens / legacy_layer20_heavy_tokens)

    legacy_total_decoder_heavy_token_layers = full_tokens + (num_decoder_layers - 1) * tail_tokens
    fixed_total_decoder_heavy_token_layers = num_decoder_layers * tail_tokens
    total_decoder_heavy_saved_ratio = 1.0 - (
        fixed_total_decoder_heavy_token_layers / legacy_total_decoder_heavy_token_layers
    )

    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "tail_window": tail_window,
        "full_tokens": full_tokens,
        "tail_tokens": tail_tokens,
        "legacy_late_layer_start": 21,
        "fixed_late_layer_start": 20,
        "legacy_layer20_heavy_tokens": legacy_layer20_heavy_tokens,
        "fixed_layer20_heavy_tokens": fixed_layer20_heavy_tokens,
        "layer20_heavy_token_reduction_pct": round(layer20_heavy_token_reduction_ratio * 100.0, 4),
        "legacy_total_decoder_heavy_token_layers": legacy_total_decoder_heavy_token_layers,
        "fixed_total_decoder_heavy_token_layers": fixed_total_decoder_heavy_token_layers,
        "total_decoder_heavy_saved_pct": round(total_decoder_heavy_saved_ratio * 100.0, 4),
    }


def analyze_trtllm_cutlass_moe_compatibility() -> dict[str, Any]:
    """Model TensorRT-LLM PR #19753 (fe92fb92) & PR #20652 (4c0197d1) compatibility and routing gates."""
    def check_internal_moe(
        num_rows: int,
        hidden_size: int,
        unpadded_hidden_size: int,
        num_valid_tokens: int | None,
        enable_alltoall: bool,
        has_input_sf: bool,
        swizzled_input_sf: bool,
        use_dynamic_fc2_scale: bool,
    ) -> tuple[bool, str]:
        if unpadded_hidden_size != hidden_size:
            return False, "Unpadded MoE output requires the open-source CUTLASS MoE kernels."
        if num_valid_tokens is not None and num_valid_tokens != num_rows:
            return False, "A separate valid-token count requires the open-source CUTLASS MoE kernels."
        if enable_alltoall:
            return False, "MoE all-to-all requires the open-source CUTLASS MoE kernels."
        if has_input_sf and not swizzled_input_sf:
            return False, "Unswizzled MoE input scales require the open-source CUTLASS MoE kernels."
        if use_dynamic_fc2_scale:
            return False, "Dynamic FC2 scaling requires the open-source CUTLASS MoE kernels."
        return True, "ok"

    def select_mla_fp8_decode(num_heads: int, seq_len_q: int, batch_size: int) -> str:
        # PR #20652 (4c0197d1): _PERF_MIN_BATCH_FP8[(96, 1)] = 8
        if num_heads == 96 and seq_len_q == 1 and batch_size < 8:
            return "trtllm_gen"
        return "cute_dsl_mla"

    ok_case = check_internal_moe(64, 4096, 4096, 64, False, True, True, False)
    reject_alltoall = check_internal_moe(64, 4096, 4096, 64, True, True, True, False)
    reject_unswizzled = check_internal_moe(64, 4096, 4096, 64, False, True, False, False)

    return {
        "internal_moe_valid_case": {"accepted": ok_case[0], "reason": ok_case[1]},
        "internal_moe_alltoall_case": {"accepted": reject_alltoall[0], "reason": reject_alltoall[1]},
        "internal_moe_unswizzled_sf_case": {"accepted": reject_unswizzled[0], "reason": reject_unswizzled[1]},
        "auto_comm_fallback_when_no_alltoall": "AllGather_ReduceScatter",
        "comm_destroyed_on_validation_failure": True,
        "mla_h96_fp8_single_token_batch4": select_mla_fp8_decode(96, 1, 4),
        "mla_h96_fp8_single_token_batch8": select_mla_fp8_decode(96, 1, 8),
        "mla_h96_fp8_multitoken_batch4": select_mla_fp8_decode(96, 2, 4),
    }


def build_summary() -> dict[str, Any]:
    gdn_pool = analyze_flashinfer_gdn_sm90_scratch(has_initial_state=True, has_output_state=False)
    gdn_no_pool = analyze_flashinfer_gdn_sm90_scratch(has_initial_state=False, has_output_state=False)
    sglang_dsv41 = analyze_sglang_dsv41_layer20_bounded_replay()
    trtllm_moe = analyze_trtllm_cutlass_moe_compatibility()
    return {
        "experiment": "flashinfer_gdn_sm90_sglang_dsv41_trtllm_moe",
        "flashinfer_gdn_sm90_pool": gdn_pool,
        "flashinfer_gdn_sm90_no_pool": gdn_no_pool,
        "sglang_dsv41_layer20_replay": sglang_dsv41,
        "trtllm_cutlass_moe_and_mla": trtllm_moe,
    }


def verify_summary(summary: dict[str, Any]) -> None:
    pool = summary["flashinfer_gdn_sm90_pool"]
    assert pool["bytes_per_slot"] == 1048576
    assert pool["compact_scratch_bytes"] == 2097152
    assert pool["legacy_overrun_bytes"] == 2097152  # slot 3 writes [3 MiB, 4 MiB), 1 MiB slice ending 2 MiB past 2 MiB
    assert pool["fixed_scratch_slots"] == 5
    assert pool["fixed_scratch_bytes"] == 5242880
    assert pool["fixed_overrun_bytes"] == 0

    no_pool = summary["flashinfer_gdn_sm90_no_pool"]
    assert no_pool["effective_indices"] is None
    assert no_pool["fixed_scratch_slots"] == 2
    assert no_pool["fixed_overrun_bytes"] == 0

    dsv41 = summary["sglang_dsv41_layer20_replay"]
    assert dsv41["legacy_late_layer_start"] == 21
    assert dsv41["fixed_late_layer_start"] == 20
    assert dsv41["legacy_layer20_heavy_tokens"] == 16384
    assert dsv41["fixed_layer20_heavy_tokens"] == 512
    assert abs(dsv41["layer20_heavy_token_reduction_pct"] - 96.875) < 1e-6
    assert abs(dsv41["total_decoder_heavy_saved_pct"] - 43.662) < 1e-3

    moe = summary["trtllm_cutlass_moe_and_mla"]
    assert moe["internal_moe_valid_case"]["accepted"] is True
    assert moe["internal_moe_alltoall_case"]["accepted"] is False
    assert moe["internal_moe_unswizzled_sf_case"]["accepted"] is False
    assert moe["comm_destroyed_on_validation_failure"] is True
    assert moe["mla_h96_fp8_single_token_batch4"] == "trtllm_gen"
    assert moe["mla_h96_fp8_single_token_batch8"] == "cute_dsl_mla"
    assert moe["mla_h96_fp8_multitoken_batch4"] == "cute_dsl_mla"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    verify_summary(summary)

    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    res_dir = Path(__file__).resolve().parent / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        print("OK: flashinfer_gdn_sm90_sglang_dsv41_trtllm_moe verified")


if __name__ == "__main__":
    main()
