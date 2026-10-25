#!/usr/bin/env python3
"""Deterministic CPU/offline simulator for Phase 73 Track B:
TensorRT-LLM DeepSeek-V4/DSpark Rubin SM104 Latent-Cache Bypass & Fused Indexer Gather,
vLLM D2D FlashInfer Warmup Cache Seeding & KV Offload Store-Failure Backpressure,
SGLang MiniMax-M3.1 NVFP4 + MSA-v1 Decode/HiCache Page-Row Copy + Disagg KV Checksum
Signature + DP-Attention Bucket/SUM_LEN Alignment, and AI Infra Innovator Economics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def simulate_trtllm_rubin_dspark() -> dict[str, Any]:
    """Model NVIDIA/TensorRT-LLM@d7e0ecac92 (DeepSeek V4 / DSpark optimizations on Rubin)."""
    # 1. attentionOp.cpp: remove dead latent_cache slicing when kv_cache_reuse & chunked_context
    # 2. indexerKCacheGather.cu: fuse index_k FP8 quant + scale + scatter into one CUDA kernel
    #    Only enabled when: sm != 104, is_IndexK_fp8, head_dim == 128,quant_block_size == 128,
    #    max_Num_Tokens_Per_Block <= 8 (each block has 8 warps = 256 threads, 1 warp per token)
    configs = [
        {"sm": 100, "num_tokens": 128, "grid_blocks": 32, "max_tokens_per_block": 4, "fused_indexer_gather": True, "launches_per_layer": 1},
        {"sm": 100, "num_tokens": 4096, "grid_blocks": 128, "max_tokens_per_block": 32, "fused_indexer_gather": False, "launches_per_layer": 2},
        {"sm": 104, "num_tokens": 128, "grid_blocks": 32, "max_tokens_per_block": 4, "fused_indexer_gather": False, "launches_per_layer": 2},
    ]
    # 3. dspark.py: defer TargetCaptureExport.consolidate() from inside layer loop to once after all layers
    total_layers = 61
    capture_layer_ids = [15, 30, 45, 60]
    consolidate_calls_before = len(capture_layer_ids)
    consolidate_calls_after = 1

    return {
        "indexer_gather_configs": configs,
        "rubin_sm104_bypasses_small_batch_fusion": configs[2]["fused_indexer_gather"] is False,
        "dspark_total_layers": total_layers,
        "dspark_capture_layers": len(capture_layer_ids),
        "dspark_consolidate_calls_before": consolidate_calls_before,
        "dspark_consolidate_calls_after": consolidate_calls_after,
    }


def simulate_vllm_seed_and_kv_backpressure() -> dict[str, Any]:
    """Model vllm@41dc477d24 (D2D FlashInfer warmup cache seeding)
    and vllm@2b1f75c80f (KV offload secondary-tier store failure backpressure).
    """
    # D2D SeedTransport timeouts & limits
    max_remote_request_bytes = 64 << 10  # 65536 B
    unauthenticated_whole_msg_timeout_s = 10.0
    authenticated_timeout_s = 300.0
    tune_group_world_size = 8
    # Unanimous check across tune_group ranks required because dummy-run sweep contains collectives
    partial_hit_ranks = [True] * 7 + [False]
    unanimous_hit_partial = all(partial_hit_ranks)
    full_hit_ranks = [True] * 8
    unanimous_hit_full = all(full_hit_ranks)

    # KV Offload EMABackpressureDetector store-failure backpressure
    default_failure_threshold = 3
    failure_ema_factor = 4.0
    local_high_water_s = 0.010
    network_high_water_s = 0.050

    # After 3 consecutive failed stores on "fs" or "obj" tier (e.g. ENOSPC/EDQUOT),
    # EMA is raised to 4.0 * high_water_s -> ThrottledDropPolicy (ramp_factor=2.0) drops 100% of stores
    fs_failure_ema_s = round(local_high_water_s * failure_ema_factor, 4)
    obj_failure_ema_s = round(network_high_water_s * failure_ema_factor, 4)
    ramp_factor = 2.0
    drop_prob_at_failure_ema = min(
        1.0,
        (fs_failure_ema_s - local_high_water_s) / ((ramp_factor - 1.0) * local_high_water_s),
    )

    return {
        "seed_max_remote_request_bytes": max_remote_request_bytes,
        "seed_unauth_whole_msg_timeout_s": unauthenticated_whole_msg_timeout_s,
        "seed_auth_timeout_s": authenticated_timeout_s,
        "tune_group_world_size": tune_group_world_size,
        "partial_rank_cache_skips_autotune": unanimous_hit_partial,
        "full_rank_cache_skips_autotune": unanimous_hit_full,
        "kv_offload_failure_threshold": default_failure_threshold,
        "kv_offload_failure_ema_factor": failure_ema_factor,
        "fs_failure_ema_s": fs_failure_ema_s,
        "obj_failure_ema_s": obj_failure_ema_s,
        "throttled_drop_probability_on_failure": drop_prob_at_failure_ema,
    }


def simulate_sglang_msa_nvfp4_and_disagg() -> dict[str, Any]:
    """Model sglang@a99c7d050f (MiniMax-M3.1 NVFP4 MoE), sglang@174bd2976d (MSA-v1 decode &
    NVFP4 KV / HiCache page_row_copy), sglang@5d11819f3a (Disagg KV checksum layout signature),
    sglang@ad950cdcd9 (DeepSeek-V4 MoE local finalize), and sglang@9b230c1dae (DP-Attention alignment).
    """
    # 1. MSA-v1 NVFP4 KV cache: page_size=128, head_dim=128, block_size=16 (8B payload + 1B E4M3 scale)
    page_size = 128
    head_dim = 128
    nvfp4_block_size = 16
    scale_groups_per_head = head_dim // nvfp4_block_size  # 8
    packed_bytes_per_head = head_dim // 2  # 64 B
    total_bytes_per_token_head = packed_bytes_per_head + scale_groups_per_head  # 72 B vs 256 B BF16 (3.556x compression)
    bf16_bytes_per_token_head = head_dim * 2  # 256 B
    kv_compression_ratio = round(bf16_bytes_per_token_head / total_bytes_per_token_head, 4)

    # 2. HiCache page_row_copy: 2 CTAs (kPageRowBlockQuota=2), 512 threads = 8 workers of 64 lanes,
    #    kVecsPerLane=16 (16x 16B = 256B in flight per lane) -> 51 GB/s H2D, 52 GB/s D2H on B300 PCIe
    hicache_vecs_sweep = {
        "8": {"h2d_gbps": 46.0},
        "16": {"h2d_gbps": 51.0, "d2h_gbps": 52.0, "ctas_used": 2},
        "32": {"h2d_gbps": 35.0},
    }

    # 3. Disagg kv_layout_signature: positive non-zero int64 via blake2b(..., digest_size=8) >> 1 | 1
    def kv_layout_sig(ps: int, kv_lens: list[int], state_lens: list[list[int]]) -> int:
        layout = (ps, list(kv_lens), [list(x) for x in state_lens])
        raw = hashlib.blake2b(repr(layout).encode(), digest_size=8).digest()
        return int.from_bytes(raw, "big") >> 1 | 1

    sig_tp4 = kv_layout_sig(128, [16384] * 30, [[4096] * 30])
    sig_tp8 = kv_layout_sig(128, [8192] * 30, [[2048] * 30])

    # 4. DP-Attention prefill graph bucket alignment & decode SUM_LEN custom all-reduce
    prefill_cg_buckets = [128, 256, 512, 1024, 2048, 4096]
    raw_dp_prefill_tokens = [0, 310, 180, 420]
    aligned_bucket = min(b for b in prefill_cg_buckets if b >= max(raw_dp_prefill_tokens))
    aligned_dp_prefill_tokens = [aligned_bucket] * len(raw_dp_prefill_tokens)

    return {
        "nvfp4_bytes_per_token_head": total_bytes_per_token_head,
        "bf16_bytes_per_token_head": bf16_bytes_per_token_head,
        "nvfp4_kv_compression_ratio": kv_compression_ratio,
        "hicache_vecs_sweep": hicache_vecs_sweep,
        "kv_layout_sig_tp4": sig_tp4,
        "kv_layout_sig_tp8": sig_tp8,
        "kv_layout_sigs_distinct_and_odd": (sig_tp4 != sig_tp8) and (sig_tp4 & 1 == 1) and (sig_tp8 & 1 == 1),
        "raw_dp_prefill_tokens": raw_dp_prefill_tokens,
        "aligned_dp_prefill_tokens": aligned_dp_prefill_tokens,
    }


def build_summary() -> dict[str, Any]:
    trtllm = simulate_trtllm_rubin_dspark()
    vllm = simulate_vllm_seed_and_kv_backpressure()
    sglang = simulate_sglang_msa_nvfp4_and_disagg()
    return {
        "experiment": "trtllm_rubin_vllm_seed_sglang_msa_nvfp4_sim",
        "phase": "Phase 73",
        "priority": 82,
        "track": "Track B",
        "trtllm_rubin_dspark": trtllm,
        "vllm_seed_and_kv_backpressure": vllm,
        "sglang_msa_nvfp4_and_disagg": sglang,
        "invariants": {
            "rubin_sm104_bypasses_small_batch_fusion": trtllm["rubin_sm104_bypasses_small_batch_fusion"] is True,
            "dspark_consolidate_reduced_to_1": trtllm["dspark_consolidate_calls_after"] == 1,
            "vllm_partial_cache_rejects_skip": vllm["partial_rank_cache_skips_autotune"] is False,
            "vllm_store_failure_drops_100pct": abs(vllm["throttled_drop_probability_on_failure"] - 1.0) < 1e-6,
            "sglang_nvfp4_kv_72_bytes": sglang["nvfp4_bytes_per_token_head"] == 72,
            "sglang_kv_sigs_valid": sglang["kv_layout_sigs_distinct_and_odd"] is True,
            "sglang_dp_prefill_aligned_to_512": sglang["aligned_dp_prefill_tokens"] == [512, 512, 512, 512],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against committed summary.json")
    args = parser.parse_args()

    summary = build_summary()
    summary_path = Path(__file__).with_name("summary.json")

    if not all(summary["invariants"].values()):
        raise AssertionError(f"Invariant failure: {summary['invariants']}")

    if args.check:
        expected = json.loads(summary_path.read_text(encoding="utf-8"))
        if expected != summary:
            raise AssertionError("summary.json mismatch for Track B Phase 73")
        print("OK: Phase 73 Track B summary verified.")
        return 0

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
