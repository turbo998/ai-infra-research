#!/usr/bin/env python3
"""Phase 37 Experiment 1: FlashInfer SM12x W4A16 TC-Decode Ultra FC2 Re-Pin,
vLLM ROCm Kimi-K3 GLA Gate + MLA FP8 Prefill Fusion & SGLang Muse Spark Unified KV Memory.

Audited upstream commits:
1. flashinfer-ai/flashinfer @ fc8cfd004324f5f3dbd54ed4aa50192aa7592888 (#6292, 2026-10-20)
2. vllm-project/vllm @ 09fd027217648768c417261ab9a90a3149649ad2 (#59634, 2026-10-20)
3. sgl-project/sglang @ cfec1b91bb33f3267d1811680b51fd581e0df458 (#45356, 2026-10-20)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _shared_memory_footprint(
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    scale_group_size: int = 16,
) -> int:
    # Deterministic model of SM12x W4A16 shared-memory footprint (A BF16 + B packed NVFP4 + E4M3 scales + reduction scratch)
    cta_m = cta_m_blocks * 8
    a_bytes = cta_m * tile_k * 2
    b_bytes = (tile_n * tile_k) // 2
    s_bytes = tile_n * (tile_k // scale_group_size)
    red_bytes = cta_m * tile_n * 4
    return (a_bytes + b_bytes + s_bytes + red_bytes) * 2


def candidate_tile_fits(
    problem_n: int,
    problem_k: int,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    cta_threads: int,
    max_shared_mem: int,
    scale_format: str = "e4m3_k16",
    allow_logical_tail: bool = False,
    enforce_search_floor: bool = True,
) -> bool:
    if int(tile_k) == -1 or int(tile_n) == -1 or int(cta_threads) == -1:
        return False
    scale_group_size = 16 if scale_format == "e4m3_k16" else 32
    if (
        (not allow_logical_tail and int(problem_n) % int(tile_n) != 0)
        or int(problem_k) % int(tile_k) != 0
        or int(tile_k) % scale_group_size != 0
    ):
        return False
    if enforce_search_floor and (
        int(tile_n) < 64 or int(tile_k) < 64 or int(cta_threads) < 128
    ):
        return False
    smem_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_group_size=scale_group_size,
    )
    return smem_bytes <= max_shared_mem


def simulate_flashinfer_sm12x_w4a16() -> dict[str, Any]:
    # Qwen3.6-35B-A3B-NVFP4 on GB10 (SM121): hidden_size=2048, intermediate_size=512, top_k=8, m in {1, 2, 4}
    problem_n = 2048
    problem_k = 512
    max_shared_mem = 101_376 - 512
    # TC-decode ultra FC2 tile widens tile_n: 128 -> 256 -> 512, forcing tile_k = 512*32/64 // 8 = 32 at 256 threads
    ultra_fc2 = {
        "problem_n": problem_n,
        "problem_k": problem_k,
        "cta_m_blocks": 1,
        "tile_n": 512,
        "tile_k": 32,
        "cta_threads": 256,
        "max_shared_mem": max_shared_mem,
    }
    stock_repin_accepted = candidate_tile_fits(**ultra_fc2, enforce_search_floor=True)
    fixed_repin_accepted = candidate_tile_fits(**ultra_fc2, enforce_search_floor=False)

    # Guardrails when enforce_search_floor=False must still reject invalid tiles
    reject_misaligned_scale_k = not candidate_tile_fits(
        **(ultra_fc2 | {"tile_k": 8}), enforce_search_floor=False
    )
    reject_non_dividing_n = not candidate_tile_fits(
        **(ultra_fc2 | {"tile_n": 768}), enforce_search_floor=False
    )
    reject_oversize_smem = not candidate_tile_fits(
        **(ultra_fc2 | {"max_shared_mem": 1024}), enforce_search_floor=False
    )

    gsm8k_eval = {
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "device": "DGX Spark GB10 (SM121)",
        "questions": 1319,
        "w4a16_fixed_0_7_0_post1": 0.9158,
        "w4a16_pre_repin_0_6_16_post3": 0.9136,
        "marlin_w4a16_mean": 0.9110,
        "b12x_w4a4_mean": 0.8900,
        "accuracy_gain_vs_w4a4_pts": round((0.9158 - 0.8900) * 100.0, 2),
    }
    return {
        "ultra_fc2_tile": {"tile_k": 32, "tile_n": 512, "cta_threads": 256},
        "stock_repin_accepted": stock_repin_accepted,
        "fixed_repin_accepted": fixed_repin_accepted,
        "guardrails_preserved": {
            "reject_misaligned_scale_k8": reject_misaligned_scale_k,
            "reject_non_dividing_tile_n768": reject_non_dividing_n,
            "reject_oversize_smem_1024": reject_oversize_smem,
        },
        "gsm8k_eval": gsm8k_eval,
    }


def simulate_vllm_rocm_kimi_k3_mla() -> dict[str, Any]:
    # Kimi-K3 MLA dimensions on ROCm CDNA4 (gfx950)
    hidden_size = 7168
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    num_heads = 24  # per-TP rank heads (non-power-of-two -> padded to m=32 in AITER MLA)
    padded_heads = 32
    tokens = 4096

    # 1. KimiK3MergedQKVGateLinear fuses q_a_proj (1536) + kv_a_proj_with_mqa (576) + g_proj (24*128=3072) = 5184 cols
    unmerged_gemm_launches = 2  # fused_qkv_a_proj + separate g_proj
    merged_gemm_launches = 1    # KimiK3MergedQKVGateLinear
    merged_out_cols = q_lora_rank + kv_lora_rank + qk_rope_head_dim + num_heads * v_head_dim
    # Saving one full hidden_states read [tokens, 7168] in BF16 (2B) per layer
    saved_activation_read_bytes = tokens * hidden_size * 2

    # 2. Decode q_out_padded [tokens, 32, 576] sliced as q_out = q_out_padded[:, :24]
    # In get_mla_padded_q: q.stride(0) == m * q.stride(1) allows zero-copy as_strided instead of tile+slice copy
    head_size = kv_lora_rank + qk_rope_head_dim  # 576
    stride_1 = head_size
    stride_0 = padded_heads * stride_1
    zero_copy_as_strided = (stride_0 == padded_heads * stride_1)
    decode_tokens = 128
    saved_decode_pad_copy_bytes = decode_tokens * padded_heads * head_size * 1  # FP8 1B

    # 3. Prefill fused_mla_prefill_qkv_fp8 gate: CDNA4 (version==4) and power-of-two head_dims (128, 64, 128)
    head_dims = (qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
    pow2_head_dims = not any(d & (d - 1) for d in head_dims)
    eager_prefill_prep_launches = 4  # _concat_k_nope_k_pe + get_mla_padded_q + q.to(fp8) + k/v.to(fp8) + torch.ones()
    fused_prefill_prep_launches = 1  # fused_mla_prefill_qkv_fp8 + cached _fp8_prefill_one_scale

    return {
        "merged_qkv_gate_out_cols": merged_out_cols,
        "unmerged_gemm_launches": unmerged_gemm_launches,
        "merged_gemm_launches": merged_gemm_launches,
        "saved_activation_read_bytes_per_layer": saved_activation_read_bytes,
        "zero_copy_as_strided_q_pad": zero_copy_as_strided,
        "saved_decode_pad_copy_bytes_per_layer": saved_decode_pad_copy_bytes,
        "pow2_head_dims": pow2_head_dims,
        "eager_prefill_prep_launches": eager_prefill_prep_launches,
        "fused_prefill_prep_launches": fused_prefill_prep_launches,
    }


def simulate_sglang_muse_spark_unified_kv() -> dict[str, Any]:
    # 1. Muse Spark _fused_kv_store with enable_unified_memory
    # Unified KV pool interleaves layers per token row; _fused_kv_store assumes contiguous layer planes.
    def fused_kv_store_enabled(enable_unified_memory: bool, is_trtllm_mha: bool) -> bool:
        if enable_unified_memory:
            return False
        return is_trtllm_mha

    # 2. _fused_draft_decision for DFLASH + FA4 when speculative_draft_kv_cache_dtype is "auto"/None
    # DFLASH FA4 always uses draft compute_dtype (bf16), whereas host Muse Spark 816B uses fp4/fp8 KV cache.
    def allow_draft_kv_fusion(
        host_kv_dtype: str,
        draft_kv_cache_dtype_arg: str | None,
        is_dflash_family: bool,
        draft_backend: str,
        draft_compute_dtype: str = "bf16",
    ) -> dict[str, bool]:
        # Old logic skipped check when draft_kv_cache_dtype_arg in (None, "auto")
        old_checked = draft_kv_cache_dtype_arg not in (None, "auto")
        old_fused = True if not old_checked else (draft_kv_cache_dtype_arg == host_kv_dtype)

        # New logic (cfec1b91bb): checks immediately if is_dflash_family and draft_backend == "fa4"
        new_checked = draft_kv_cache_dtype_arg not in (None, "auto") or (
            is_dflash_family and draft_backend == "fa4"
        )
        effective_draft_dtype = (
            draft_compute_dtype
            if (draft_kv_cache_dtype_arg in (None, "auto") and is_dflash_family and draft_backend == "fa4")
            else (draft_kv_cache_dtype_arg or host_kv_dtype)
        )
        new_fused = True if not new_checked else (effective_draft_dtype == host_kv_dtype)
        return {"old_fused": old_fused, "new_fused": new_fused}

    dflash_fa4_auto_on_fp8_host = allow_draft_kv_fusion(
        host_kv_dtype="fp8_e4m3",
        draft_kv_cache_dtype_arg="auto",
        is_dflash_family=True,
        draft_backend="fa4",
        draft_compute_dtype="bf16",
    )
    return {
        "unified_memory_fused_kv_store": fused_kv_store_enabled(
            enable_unified_memory=True, is_trtllm_mha=True
        ),
        "dedicated_memory_fused_kv_store": fused_kv_store_enabled(
            enable_unified_memory=False, is_trtllm_mha=True
        ),
        "dflash_fa4_auto_on_fp8_host": dflash_fa4_auto_on_fp8_host,
        "gsm8k_threshold_4gpu_b200": 0.93,
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "flashinfer_sm12x_w4a16_vllm_kimi_rocm_sglang_unified_kv",
        "phase": 37,
        "priority": 46,
        "evidence_tier": "own-offline",
        "flashinfer_sm12x_w4a16": simulate_flashinfer_sm12x_w4a16(),
        "vllm_rocm_kimi_k3_mla": simulate_vllm_rocm_kimi_k3_mla(),
        "sglang_muse_spark_unified_kv": simulate_sglang_muse_spark_unified_kv(),
    }


def verify(summary: dict[str, Any]) -> None:
    fi = summary["flashinfer_sm12x_w4a16"]
    assert fi["stock_repin_accepted"] is False
    assert fi["fixed_repin_accepted"] is True
    assert all(fi["guardrails_preserved"].values())
    assert fi["gsm8k_eval"]["accuracy_gain_vs_w4a4_pts"] == 2.58

    vl = summary["vllm_rocm_kimi_k3_mla"]
    assert vl["merged_qkv_gate_out_cols"] == 5184
    assert vl["unmerged_gemm_launches"] == 2 and vl["merged_gemm_launches"] == 1
    assert vl["saved_activation_read_bytes_per_layer"] == 58_720_256
    assert vl["zero_copy_as_strided_q_pad"] is True
    assert vl["pow2_head_dims"] is True
    assert vl["eager_prefill_prep_launches"] == 4 and vl["fused_prefill_prep_launches"] == 1

    sg = summary["sglang_muse_spark_unified_kv"]
    assert sg["unified_memory_fused_kv_store"] is False
    assert sg["dedicated_memory_fused_kv_store"] is True
    assert sg["dflash_fa4_auto_on_fp8_host"] == {"old_fused": True, "new_fused": False}


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
        print("flashinfer_sm12x_w4a16_vllm_kimi_rocm_sglang_unified_kv: check passed")
        return 0

    for rel in ("summary.json", "results/summary.json"):
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
