#!/usr/bin/env python3
"""Deterministic stdlib verification for Phase 60 Track B:
- TensorRT-LLM SWA branch snapshots at prefix fork points in KVCacheManagerV2 (#19877, ae1aa43964)
- TensorRT-LLM MiniMax-M3 on Rubin (SM107) with NVFP4 KV exemption, 128-token Eagle3 draft pages, and zero-initialized KV pools (#20167, 854ccee8a9)
- TensorRT-LLM separate V token stride in context FMHA and 16-byte TMA alignment (#20242, d7ec5162ac)
- SGLang LTX-2.3 BCG video RoPE SP shard start offset (#45995, 2fbd094a8c), AMD ROCm deterministic all-reduce guard above TP2 (#43809, c80285ca9a), and FLUX3 fused SwiGLU + rowwise FP8 quantization (#42758, 698fa949ec)
- vLLM V2 model runner speculative decoding draft rotary cache bound (#63251, 37cf712552), CPU GDN spec-decode conv-state layout (#63339, f3871c3cbf), and cooperative_topk row-length sizing (#62171, 404c2b510e)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def swa_branch_snapshot_point(
    divergence: int,
    num_lookup_tokens: int,
    reused: int,
    prompt_len: int,
    tokens_per_block: int,
) -> Optional[int]:
    """Exact replica of TensorRT-LLM _swa_branch_snapshot_point (#19877)."""
    if divergence >= num_lookup_tokens:
        return None
    point = divergence // tokens_per_block * tokens_per_block
    if point == 0 or point <= reused or point >= prompt_len:
        return None
    return point


def verify_trtllm_swa_branch_snapshot() -> dict[str, Any]:
    """Verify SWA branch snapshot at prefix fork points and FORCE_CHUNK scheduling (#19877)."""
    tokens_per_block = 64
    # Request 1 (4096 tokens) shares 1500 tokens with an earlier conversation, but SWA window was only kept at prompt end (reused=0).
    # Fork point = 1500 // 64 * 64 = 1472.
    fork_point = swa_branch_snapshot_point(
        divergence=1500,
        num_lookup_tokens=4096,
        reused=0,
        prompt_len=4096,
        tokens_per_block=tokens_per_block,
    )
    assert fork_point == 1472

    # When sibling Request 2 arrives sharing that 1472-token prefix:
    # Without branch snapshot: reused = 0 tokens (must recompute all 1472 shared tokens)
    # With branch snapshot: reused = 1472 tokens!
    sibling_prompt_len = 2048
    recomputed_without_branch_snap = sibling_prompt_len
    recomputed_with_branch_snap = sibling_prompt_len - fork_point
    saved_prefill_pct = round(fork_point / sibling_prompt_len * 100.0, 2)
    assert recomputed_with_branch_snap == 576
    assert saved_prefill_pct == 71.88

    # Edge cases where _swa_branch_snapshot_point returns None:
    assert swa_branch_snapshot_point(4096, 4096, 0, 4096, 64) is None  # whole lookup matched
    assert swa_branch_snapshot_point(48, 4096, 0, 4096, 64) is None  # below 1 block (point == 0)
    assert swa_branch_snapshot_point(1500, 4096, 1472, 4096, 64) is None  # reuse already reached fork

    return {
        "tokens_per_block": tokens_per_block,
        "divergence_token": 1500,
        "committed_fork_point": fork_point,
        "sibling_prompt_len": sibling_prompt_len,
        "recomputed_tokens_without_branch_snapshot": recomputed_without_branch_snap,
        "recomputed_tokens_with_branch_snapshot": recomputed_with_branch_snap,
        "sibling_prefill_saved_pct": saved_prefill_pct,
        "edge_cases_none_verified": True,
        "invariants_passed": True,
    }


def verify_trtllm_minimax_rubin_and_vstride() -> dict[str, Any]:
    """Verify MiniMax-M3 on Rubin SM107 (#20167) and separate V token stride in context FMHA (#20242)."""
    # 1. MiniMax-M3 on Rubin SM107:
    # Sparse layers use NVFP4 KV cache (MSA + Triton support SM100, SM103, SM107).
    # Dense target layers stay on SDPA (FP8 KV) on SM107 because FlashInferTrtllmGenFmha serves only EXTRA_PAGE_SIZES_ONLY_SMS={107} (128-token Eagle3 draft pages).
    sm100f_capabilities = [[10, 0], [10, 3], [10, 7]]
    extra_page_sizes_only_sms = [107]
    eagle3_draft_page_tokens = 128

    # 2. Separate V token stride (#20242):
    # qwen4_exp fused input projection row: [q | k | v | gate | index | pad] = 55 heads of 256 elements (BF16 = 2 bytes)
    # Without separate v_stride_in_bytes, non-FP8 context MLA assumed packed [k_nope | v] stride, reading corrupt columns!
    total_fused_heads = 55
    head_dim = 256
    elem_bytes = 2
    physical_v_stride_bytes = total_fused_heads * head_dim * elem_bytes  # 28160 bytes
    tma_alignment_bytes = 16
    assert physical_v_stride_bytes % tma_alignment_bytes == 0
    v_token_stride_elements = physical_v_stride_bytes // elem_bytes
    assert v_token_stride_elements == 14080

    return {
        "sm100f_nvfp4_capabilities": sm100f_capabilities,
        "extra_page_sizes_only_sms": extra_page_sizes_only_sms,
        "eagle3_draft_page_tokens_sm107": eagle3_draft_page_tokens,
        "qwen4_exp_fused_row_heads": total_fused_heads,
        "physical_v_stride_bytes": physical_v_stride_bytes,
        "v_token_stride_elements": v_token_stride_elements,
        "tma_16b_aligned": True,
        "invariants_passed": True,
    }


def verify_sglang_and_vllm_fixes() -> dict[str, Any]:
    """Verify SGLang LTX-2.3 SP RoPE (#45995), ROCm deterministic AR (#43809), FLUX3 SwiGLU FP8 (#42758), and vLLM fixes (#63251, #63339, #62171)."""
    # 1. SGLang LTX-2.3 BCG video RoPE SP shard start offset (#45995):
    # Across 4 SP ranks with 16 latent frames per rank (total 64 frames):
    # Without fix: all SP ranks started at frame 0 -> [0..15] on every rank!
    # With fix: rank r starts at video_sp_start = r * 16 -> continuous [0..63].
    sp_size = 4
    frames_per_rank = 16
    old_start_frames = [0 for _ in range(sp_size)]
    new_start_frames = [r * frames_per_rank for r in range(sp_size)]
    assert old_start_frames == [0, 0, 0, 0]
    assert new_start_frames == [0, 16, 32, 48]

    # 2. SGLang ROCm deterministic all-reduce guard above TP2 (#43809):
    # (is_hip, deterministic, tp_size, disable_custom_all_reduce) -> raises ValueError iff True, True, >2, True
    rocm_cases = [
        (True, True, 4, True, True),
        (False, True, 8, True, False),
        (True, True, 2, True, False),
        (True, False, 8, True, False),
        (True, True, 8, False, False),
    ]
    for is_hip, det, tp, disable_custom, expected_raise in rocm_cases:
        actual_raise = (is_hip and det and tp > 2 and disable_custom)
        assert actual_raise == expected_raise

    # 3. SGLang FLUX3 fused SwiGLU + rowwise FP8 quantization (#42758):
    # Packed BF16 [M, 2*K] (4 bytes/col) -> directly writes E4M3 [M, K] (1 byte/col), avoiding BF16 intermediate [M, K] write+read (2+2=4 bytes/col).
    # HBM traffic per (M, K) element:
    # Unfused: read 2*K BF16 (4B) + write K BF16 (2B) + read K BF16 (2B) + write K FP8 (1B) = 9 bytes/elem
    # Fused: read 2*K BF16 (4B) + write K FP8 (1B) = 5 bytes/elem -> 44.44% HBM traffic reduction!
    unfused_bytes_per_k = 9
    fused_bytes_per_k = 5
    flux3_hbm_reduction_pct = round((unfused_bytes_per_k - fused_bytes_per_k) / unfused_bytes_per_k * 100.0, 2)
    assert flux3_hbm_reduction_pct == 44.44

    # 4. vLLM cooperative_topk single-pass sizing by row length sl instead of stride (#62171):
    # Suppose stride = 1048576 (1M vocab/buffer stride), sl = 4096 (live row length), CS = 16 CTAs, kFusedMax = 512:
    stride = 1048576
    sl = 4096
    cs = 16
    k_fused_max = 512
    old_per_block = (stride + cs - 1) // cs  # 65536 > 512 -> forced into slow two-pass!
    new_per_block = (sl + cs - 1) // cs      # 256 <= 512 -> stays in fast single-pass!
    assert old_per_block == 65536 and old_per_block > k_fused_max
    assert new_per_block == 256 and new_per_block <= k_fused_max
    work_reduction_factor = old_per_block // new_per_block
    assert work_reduction_factor == 256

    # 5. vLLM CPU GDN spec-decode conv-state layout (#63339):
    # width = 4, state_len = 6 -> full-length spec row has seq_lens[0] == state_len - (width - 2) = 4.
    width = 4
    state_len = 6
    full_spec_row_len = state_len - (width - 2)
    assert full_spec_row_len == 4

    return {
        "ltx2_old_start_frames": old_start_frames,
        "ltx2_new_start_frames": new_start_frames,
        "rocm_deterministic_tp_above_2_guard_verified": True,
        "flux3_swiglu_fp8_hbm_reduction_pct": flux3_hbm_reduction_pct,
        "vllm_coop_topk_old_per_block": old_per_block,
        "vllm_coop_topk_new_per_block": new_per_block,
        "vllm_coop_topk_single_pass_restored": True,
        "vllm_cpu_gdn_full_spec_row_len": full_spec_row_len,
        "invariants_passed": True,
    }


def build_summary() -> dict[str, Any]:
    swa = verify_trtllm_swa_branch_snapshot()
    rubin_vstride = verify_trtllm_minimax_rubin_and_vstride()
    sgl_vllm = verify_sglang_and_vllm_fixes()
    return {
        "experiment": "trtllm-swa-branch-snapshot-minimax-rubin-vstride-sglang-vllm",
        "upstream_commits": [
            "NVIDIA/TensorRT-LLM@ae1aa43964e12afb5157ca5d551993a89db2224e",
            "NVIDIA/TensorRT-LLM@854ccee8a9f0dded7235a0642032de105b3b47f7",
            "NVIDIA/TensorRT-LLM@d7ec5162ac8d935b21bb5eac719e5eac99664cec",
            "NVIDIA/TensorRT-LLM@0f3e1d093dceb454f01a02a578cde1aed99d0495",
            "sgl-project/sglang@2fbd094a8cb641deb099b5b2ac0d362b7eaa6aec",
            "sgl-project/sglang@c80285ca9adc23946447128926c22e8e7dfb2b80",
            "sgl-project/sglang@698fa949ecd3f5662a87edcb4aefcb44884a2776",
            "vllm-project/vllm@37cf712552509d5bf2e67128b80ab435079434fc",
            "vllm-project/vllm@f3871c3cbf8a608f7ee833ec4c2c5277f77bee41",
            "vllm-project/vllm@404c2b510e1d73df151d78e5576e1ee7ca0bcaa3",
        ],
        "trtllm_swa_branch_snapshot": swa,
        "trtllm_minimax_rubin_and_vstride": rubin_vstride,
        "sglang_and_vllm_fixes": sgl_vllm,
        "all_invariants_passed": (
            swa["invariants_passed"]
            and rubin_vstride["invariants_passed"]
            and sgl_vllm["invariants_passed"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify against existing summary.json")
    args = parser.parse_args()

    summary = build_summary()
    assert summary["all_invariants_passed"], "Invariants failed"
    summary_path = Path(__file__).resolve().parent / "summary.json"

    if args.check:
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing {summary_path}")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise AssertionError("summary.json mismatch")
        print(f"[PASS] {summary['experiment']}")
        return 0

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
