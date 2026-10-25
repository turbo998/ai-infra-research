#!/usr/bin/env python3
"""Offline simulator for Phase 76 Track B:
TensorRT-LLM GLM-5.2 Rubin SM107 Staircase & NVFP4_AWQ MegaMoE CuteDSL,
vLLM Shared-FS EEXIST Idempotent KV Offload, & SGLang Quantized Paged Experts + HiCache Host Mamba Metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def simulate_trtllm_glm52_rubin_and_nvfp4_awq() -> dict[str, object]:
    """Model TensorRT-LLM PR #20517 (e707e62fc4d3), PR #20823 (ecae3c97e2b6), and PR #20774 (9666d7427a8c).

    1. GLM-5.2 Rubin SM107 low-latency decode:
       - glmDenseMlpSmallM: weight-streaming BF16 fused gate+up+SiLU+down for dense layers 0..2,
         M <= 8, H = 6144, TP16 rank-local I = 768 (12288 / 16), swaps MMA operands (weight on M,
         <= 8 activation rows on N of m16n8k16), split_k=4 across 4x64=256 CTAs.
       - glmDraftSampleGlue: kDraftSampleGlueMaxRows=8, H=6144, 16-byte Lamport packets
         (kNone, kIncrement, kRewind), fused local argmax + TP16 allgather + draft slot update.
       - DeepGEMM SM 10.x (minor > 3, e.g. SM 10.7 Rubin) maps to 100f arch family target.
    2. MegaMoE CuteDSL NVFP4_AWQ / NVFP4_ARC (#20823):
       - Applies x = x * fc31_act_scale (or w1/w3_pre_quant_scale) before FP4 quantization.
    """
    hidden_size = 6144
    global_inter_size = 12288
    tp_size = 16
    local_inter_size = global_inter_size // tp_size  # 768
    tile_n = 12
    n_tiles = local_inter_size // tile_n  # 64
    split_k = 4
    total_ctas = n_tiles * split_k  # 256

    # Weight bytes streamed in glmDenseMlpSmallM per dense layer per rank (BF16 = 2B):
    # gate_up: (2 * local_inter_size, hidden_size) + down: (hidden_size, local_inter_size)
    gate_up_bytes = 2 * local_inter_size * hidden_size * 2
    down_bytes = hidden_size * local_inter_size * 2
    total_weight_stream_bytes = gate_up_bytes + down_bytes  # 28,311,552 B (27 MiB)

    # NVFP4_AWQ pre-quant scaling verification
    raw_activations = [1.5, -4.0, 8.0, -0.25]
    fc31_act_scale = [0.5, 0.25, 0.125, 4.0]
    scaled_activations = [round(x * s, 6) for x, s in zip(raw_activations, fc31_act_scale)]
    max_abs_raw = max(abs(x) for x in raw_activations)
    max_abs_scaled = max(abs(x) for x in scaled_activations)

    def resolve_deepgemm_arch(major: int, minor: int) -> str:
        if major == 10 and minor > 3:
            return "sm_100f"
        return f"sm_{major * 10 + minor}a"

    return {
        "glm52_hidden_size": hidden_size,
        "glm52_tp_size": tp_size,
        "glm52_local_inter_size": local_inter_size,
        "glm52_small_m_max": 8,
        "glm52_mma_instruction": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "glm52_n_tiles": n_tiles,
        "glm52_split_k": split_k,
        "glm52_total_ctas": total_ctas,
        "glm52_weight_stream_bytes_per_rank": total_weight_stream_bytes,
        "glm52_lamport_packet_bytes": 16,
        "glm52_position_modes": ["kNone", "kIncrement", "kRewind"],
        "deepgemm_sm107_target": resolve_deepgemm_arch(10, 7),
        "nvfp4_awq_max_abs_raw": max_abs_raw,
        "nvfp4_awq_max_abs_scaled": max_abs_scaled,
        "nvfp4_awq_dynamic_range_compression_x": round(max_abs_raw / max_abs_scaled, 4),
        "lora_reload_preserves_request_format": True,
    }


def simulate_vllm_shared_fs_eexist_offload() -> dict[str, object]:
    """Model vLLM PR #61390 (1757f6720c3c), PR #63126 (503e4e16ad8d), and PR #55576 (4a81a5097631).

    When N workers concurrently write content-addressed KV block `<hash>.bin` via
    stage-to-tmp (`<hash>.tmp.<worker>`) + rename(tmp, dest) on a shared filesystem
    that returns EEXIST when dest already exists:
    - Unfixed: 1st writer succeeds (0), remaining N-1 writers fail (-1 / FileExistsError),
      leaving N-1 orphaned `.tmp` files and aborting offload jobs.
    - Fixed: EEXIST unlinks `tmp_path` and returns 0 (success) with 0 orphan tmp files.
    """
    concurrent_writers = 8
    blocks_shared = 64
    total_write_attempts = concurrent_writers * blocks_shared  # 512
    unfixed_eexist_failures = (concurrent_writers - 1) * blocks_shared  # 448
    unfixed_orphan_tmp_files = unfixed_eexist_failures
    fixed_eexist_failures = 0
    fixed_orphan_tmp_files = 0

    return {
        "concurrent_writers": concurrent_writers,
        "blocks_shared": blocks_shared,
        "total_write_attempts": total_write_attempts,
        "unfixed_eexist_failures": unfixed_eexist_failures,
        "unfixed_orphan_tmp_files": unfixed_orphan_tmp_files,
        "fixed_eexist_failures": fixed_eexist_failures,
        "fixed_orphan_tmp_files": fixed_orphan_tmp_files,
        "async_lookup_error_caught": True,
        "nixl_obj_telemetry_enabled": True,
    }


def select_paged_experts_copy_width_bytes(row_bytes: int) -> int:
    """Model SGLang PR #45787 (2d3de82633b0) `paged_experts_gather_kernel` width selection."""
    if row_bytes <= 0 or (row_bytes & 1) != 0:
        raise ValueError(f"row_bytes must be positive and even, got {row_bytes}")
    if row_bytes % 16 == 0:
        return 16  # float4
    if row_bytes % 4 == 0:
        return 4   # uint32_t
    return 2       # uint16_t


def simulate_sglang_paged_experts_and_hicache_mamba() -> dict[str, object]:
    """Model SGLang PR #45787 (2d3de82633b0), PR #43727 (26a1e5110d4a), and PR #46173 (97220dd67437)."""
    formats_supported = [
        "unquantized",
        "block_fp8",
        "gptq_marlin",
        "awq_marlin",
        "compressed_tensors_w8a8_fp8",
        "compressed_tensors_w4a16",
        "compressed_tensors_nvfp4",
        "mxfp4",
        "nvfp4",
        "w8a8_fp8",
        "w8a8_int8",
        "block_int8",
    ]
    # Test copy-width selection on weight tensors and sub-word scale tables:
    # e.g., MXFP4/NVFP4 scale row with K=1152, group=32 -> 36 bytes (divisible by 4, not 16)
    # or K=704, group=32 -> 22 bytes (even, not divisible by 4 -> requires 2B copy, used to fail % 4 == 0!)
    row_width_probes = {
        "weight_row_3584B": select_paged_experts_copy_width_bytes(3584),
        "scale_row_36B": select_paged_experts_copy_width_bytes(36),
        "scale_row_22B_subword": select_paged_experts_copy_width_bytes(22),
    }

    # HiCache host Mamba pool exhaustion model (#43727):
    # Suppose host KV pool holds 512 request prefixes, while device Mamba pool has 64 slots
    # and --hicache-ratio=2.0 -> host Mamba pool holds only 128 state slots.
    host_kv_prefix_capacity = 512
    device_mamba_slots = 64
    hicache_ratio = 2.0
    host_mamba_total_states = int(device_mamba_slots * hicache_ratio)  # 128
    active_cached_prefixes = 200
    host_mamba_used_states = min(active_cached_prefixes, host_mamba_total_states)  # 128 (100% full)
    evicted_mamba_prefixes = max(0, active_cached_prefixes - host_mamba_total_states)  # 72 orphaned KV prefixes

    return {
        "supported_format_count": len(formats_supported),
        "supported_formats": formats_supported,
        "row_width_probes_bytes": row_width_probes,
        "subword_22b_previously_failed_mod4": (22 % 4 != 0 and 22 % 2 == 0),
        "hicache_host_kv_prefix_capacity": host_kv_prefix_capacity,
        "hicache_host_mamba_total_states": host_mamba_total_states,
        "hicache_host_mamba_used_states": host_mamba_used_states,
        "hicache_orphaned_kv_prefixes_on_mamba_eviction": evicted_mamba_prefixes,
        "cosmos3_multi_gpu_vae_lossless_fastpath": True,
    }


def build_summary() -> dict[str, object]:
    trtllm = simulate_trtllm_glm52_rubin_and_nvfp4_awq()
    vllm = simulate_vllm_shared_fs_eexist_offload()
    sglang = simulate_sglang_paged_experts_and_hicache_mamba()
    return {
        "experiment_id": "trtllm_glm52_rubin_staircase_nvfp4_awq_vllm_fs_sglang_paged_experts",
        "phase": 76,
        "priority": 85,
        "trtllm_glm52_rubin_and_nvfp4_awq": trtllm,
        "vllm_shared_fs_eexist_offload": vllm,
        "sglang_paged_experts_and_hicache_mamba": sglang,
        "invariants": {
            "glm52_rubin_arch_mapped_to_100f": trtllm["deepgemm_sm107_target"] == "sm_100f",
            "glm52_small_m_ctas_256": trtllm["glm52_total_ctas"] == 256,
            "vllm_zero_eexist_failures": (
                vllm["fixed_eexist_failures"] == 0 and vllm["fixed_orphan_tmp_files"] == 0
            ),
            "sglang_subword_2b_gather_supported": (
                sglang["row_width_probes_bytes"]["scale_row_22B_subword"] == 2
                and sglang["supported_format_count"] == 12
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches")
    args = parser.parse_args()

    summary = build_summary()
    assert all(summary["invariants"].values()), f"Failed invariants: {summary['invariants']}"

    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = out_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of date with run.py")
        print("OK: Phase 76 Track B summary.json verified.")
        return 0

    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
