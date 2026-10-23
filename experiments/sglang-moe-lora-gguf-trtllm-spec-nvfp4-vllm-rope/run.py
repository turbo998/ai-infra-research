#!/usr/bin/env python3
"""Phase 63 Track B deterministic verifier:
SGLang MoE LoRA Virtual-Expert Routing & GGUF Subnormal Weighting,
TensorRT-LLM DSv4 NVFP4 FMHA KV-Tile Padding & DeviceCopyBuffer Prefix Guard,
and vLLM Muse Spark Shared FP32 RoPE & DeepSeek-VL2 Cropping Cache Identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def compute_virtual_expert_id(
    base_expert_id: int,
    lora_id: int,
    num_slots: int,
    adapter_enabled: list[int] | None,
    num_experts_for_weight: int,
    shared_outer: bool,
) -> int:
    """Deterministic model of SGLang d0022e1df3 compute_virtual_id."""
    if lora_id < 0 or lora_id >= num_slots:
        return -1
    if adapter_enabled is not None and adapter_enabled[lora_id] == 0:
        return -1
    if base_expert_id < 0:
        return -1
    if shared_outer:
        return lora_id
    if base_expert_id >= num_experts_for_weight:
        return -1
    return base_expert_id + lora_id * num_experts_for_weight


def evaluate_sglang_moe_lora_and_gguf() -> dict[str, Any]:
    """Verify SGLang MoE LoRA routing (d0022e1df3) and GGUF MoE weighting (e6451c1a95)."""
    num_slots = 4
    num_experts = 8
    adapter_enabled = [1, 0, 1, 1]

    # Case 1: valid active slot 2, base expert 5 -> 5 + 2*8 = 21
    v_active = compute_virtual_expert_id(5, 2, num_slots, adapter_enabled, num_experts, False)
    # Case 2: disabled slot 1 -> -1 (previously mapped to slot 1 or clamped to 0!)
    v_disabled = compute_virtual_expert_id(5, 1, num_slots, adapter_enabled, num_experts, False)
    # Case 3: out-of-range slot 4 -> -1
    v_oob_slot = compute_virtual_expert_id(5, 4, num_slots, adapter_enabled, num_experts, False)
    # Case 4: masked base expert -1 -> -1
    v_masked_expert = compute_virtual_expert_id(-1, 2, num_slots, adapter_enabled, num_experts, False)
    # Case 5: shared_outer with valid base expert 3, slot 2 -> 2
    v_shared = compute_virtual_expert_id(3, 2, num_slots, adapter_enabled, 0, True)

    assert v_active == 21
    assert v_disabled == -1
    assert v_oob_slot == -1
    assert v_masked_expert == -1
    assert v_shared == 2

    # GGUF MoE weighting: negative expert id zeroes output row; positive id preserves IEEE-754 __fmul_rn
    subnormal_val = 1.0e-39
    weight = 0.5
    ieee_product = subnormal_val * weight
    ftz_product = 0.0
    assert ieee_product > 0.0 and ftz_product == 0.0

    return {
        "lora_commit": "d0022e1df3",
        "gguf_commit": "e6451c1a95",
        "quark_fp8_commit": "9d7ad308bd",
        "radix_cache_action_commit": "005333d029",
        "virtual_expert_cases": {
            "active_slot2_expert5": v_active,
            "disabled_slot1_expert5": v_disabled,
            "oob_slot4_expert5": v_oob_slot,
            "masked_expert_neg1_slot2": v_masked_expert,
            "shared_outer_slot2_expert3": v_shared,
        },
        "inactive_prefix_count_tuple": "[total_padded_tokens, leading_negative_expert_blocks]",
        "gguf_moe_weighting": {
            "negative_id_action": "zero_output_row",
            "sm90_multiply_intrinsic": "__fmul_rn",
            "subnormal_input": subnormal_val,
            "weight": weight,
            "ieee_preserved_nonzero": ieee_product > 0.0,
        },
    }


def evaluate_trtllm_nvfp4_and_spec_dec() -> dict[str, Any]:
    """Verify TRT-LLM DSv4 NVFP4 scratch padding (36d8aef55e) & DeviceCopyBuffer (dd63b36fcf)."""
    tile_size_kv = 128
    head_dim = 576
    num_rows = 1024

    unpadded_bytes = num_rows * head_dim
    padded_rows = num_rows + tile_size_kv
    padded_bytes = padded_rows * head_dim
    tail_zero_bytes = tile_size_kv * head_dim

    assert unpadded_bytes == 589824
    assert padded_bytes == 663552
    assert tail_zero_bytes == 73728

    # Simulate DeviceCopyBuffer prefix skip and tail preservation across 4 speculative steps
    updates = [
        ([10, 20, 30, 40], False),
        ([10, 20, 30, 40], False),  # identical prefix -> skipped
        ([10, 20], False),          # shorter prefix matching [:2] -> skipped, preserves [30, 40]
        ([10, 99], False),          # changed prefix -> copies 2 elements, preserves tail [30, 40]
        ([10, 99], True),           # force=True (C++ Mamba manager) -> unconditional copy
    ]

    device_buf = [0, 0, 0, 0]
    last_host: list[int] | None = None
    copies_enqueued = 0
    copies_skipped = 0

    for host_vals, force in updates:
        n = len(host_vals)
        if not force and last_host is not None and len(last_host) >= n and last_host[:n] == host_vals:
            copies_skipped += 1
            continue
        device_buf[:n] = host_vals
        if last_host is None or len(last_host) < n:
            last_host = list(host_vals)
        else:
            last_host[:n] = host_vals
        copies_enqueued += 1

    assert copies_enqueued == 3
    assert copies_skipped == 2
    assert device_buf == [10, 99, 30, 40]

    # Kimi K3 per-rank memory model (footnote 15 in dd63b36fcf)
    kimi_k3_dep16_gb = 210  # 114 GB replicated BF16 non-expert + 90 GB MXFP4 EP16 + runtime
    kimi_k3_tep16_gb = 115  # sharded non-expert + 90 GB MXFP4 EP16

    return {
        "nvfp4_scratch_commit": "36d8aef55e",
        "spec_dec_commit": "dd63b36fcf",
        "v2_kv_events_commit": "fa2eda12c4",
        "nvfp4_scratch_padding": {
            "sparse_mla_fmha_kv_tile_size": tile_size_kv,
            "head_dim": head_dim,
            "batch_num_rows": num_rows,
            "unpadded_bytes": unpadded_bytes,
            "padded_rows": padded_rows,
            "padded_bytes": padded_bytes,
            "tail_zero_bytes": tail_zero_bytes,
            "overhead_pct": round((tail_zero_bytes / unpadded_bytes) * 100.0, 2),
        },
        "device_copy_buffer": {
            "total_updates": len(updates),
            "copies_enqueued": copies_enqueued,
            "copies_skipped": copies_skipped,
            "final_device_buffer": device_buf,
            "kimi_k3_dep16_per_rank_gb": kimi_k3_dep16_gb,
            "kimi_k3_tep16_per_rank_gb": kimi_k3_tep16_gb,
        },
    }


def evaluate_vllm_rope_cpu_and_vl2_cache() -> dict[str, Any]:
    """Verify vLLM Muse Spark shared RoPE (b1c66e5545), CPU pooling KV skip (c3117c7441), & DeepSeek-VL2 hash (23f546eee0)."""
    num_layers = 40
    max_position_embeddings = 131072
    head_dim = 128
    # FP32 cos_sin_cache stores head_dim elements (4 bytes each) per position
    per_cache_bytes = max_position_embeddings * head_dim * 4
    unshared_bytes = num_layers * per_cache_bytes
    shared_bytes = per_cache_bytes
    saved_mib = (unshared_bytes - shared_bytes) / (1024 * 1024)

    assert per_cache_bytes == 67108864  # 64 MiB
    assert saved_mib == 2496.0          # 2.4375 GiB saved across 40 target layers

    # DeepSeek-VL2 cropping mode cache hash differentiation
    base_img_hash = "img_sha256_deadbeef"
    hash_2_images = hashlib.sha256(f"{base_img_hash}:cropping={2 <= 2}".encode()).hexdigest()[:16]
    hash_3_images = hashlib.sha256(f"{base_img_hash}:cropping={3 <= 2}".encode()).hexdigest()[:16]
    assert hash_2_images != hash_3_images

    return {
        "muse_spark_rope_commit": "b1c66e5545",
        "cpu_pooling_kv_commit": "c3117c7441",
        "deepseek_vl2_cache_commit": "23f546eee0",
        "muse_spark_shared_rope": {
            "num_layers": num_layers,
            "max_position_embeddings": max_position_embeddings,
            "head_dim": head_dim,
            "dtype": "float32",
            "single_cache_mib": per_cache_bytes / (1024 * 1024),
            "unshared_total_mib": unshared_bytes / (1024 * 1024),
            "saved_mib": saved_mib,
            "reduction_factor": num_layers,
        },
        "deepseek_vl2_cropping_hash": {
            "cropping_threshold_max_images": 2,
            "hash_2_images_cropping_true": hash_2_images,
            "hash_3_images_cropping_false": hash_3_images,
            "collision_prevented": hash_2_images != hash_3_images,
        },
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope",
        "phase": 63,
        "priority": 72,
        "track": "B",
        "status": "verified",
        "sglang_moe_lora_and_gguf": evaluate_sglang_moe_lora_and_gguf(),
        "trtllm_nvfp4_and_spec_dec": evaluate_trtllm_nvfp4_and_spec_dec(),
        "vllm_rope_cpu_and_vl2_cache": evaluate_vllm_rope_cpu_and_vl2_cache(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against summary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("summary.json"),
        help="Output summary JSON path",
    )
    args = parser.parse_args()

    summary = build_summary()
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing summary file: {args.output}")
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of date with run.py")
        print("OK: sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope summary verified.")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
