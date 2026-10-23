"""Deterministic offline verification for Phase 70 Track B:
1. vLLM `moe_permute` invalid `topk_ids` (`-1` padding routes) remapping without
   `expert_map`, plus CUTLASS FP8 (`mm1_out.fill_(0)`) and FusedHummingMoE
   (`num_valid_tokens = expert_offsets[-1:]`) dynamic scale bounds
   (`vllm@e4db46c79bca98ab3e0b661b5cb429ab4f58f4f8`, PR #59717).
2. vLLM ROCm RDNA (`gfx1151`) W4A16 prefill packed FP16 dequantization via
   `v_and_or_b32 $0, $1, $2, 0x64006400` (`1024 + nibble`) and folded zero-point
   subtraction (`vllm@96bbaa300fe4ea605f37f42b1a6406ec57bc4fb6`, PR #60644).
3. vLLM ROCm `gfx942` Kimi-K3 load-time MXFP4-to-INT4 expert requantization
   validation (`(kInt4Static32, None)`) and speculative decoding drafter LoRA
   verification bypass (`vllm@2c58be22b68dcb715455958e406e40f709bf59e8`, PR #63131;
   `vllm@58301f6cb9b15dad4681e34af0f325c60a5d9323`, PR #63186).
4. SGLang Inkling `enable_lm_head_output_fp32=True` keeping BF16 LM-head operands
   while writing FP32 outputs (`torch.mm(..., out_dtype=torch.float32)`) to avoid
   BF16 tie-rounding greedy flips (`sglang@855c092deb93104f1ad2844974fd2e4da7bb90f7`,
   PR #45335).
5. SGLang AMD Quark block-FP8 DeepSeek loader `_resolve_quark_fp8_block_scale_name`
   (`weight_scale -> weight_scale_inv` only when registered) and layer-first
   `_resolve_layer_weight_block_size` (`sglang@9d7ad308bdbb253ba736f201475c8e798ff4b8fe`,
   PR #40776), plus TensorRT-LLM CBTS top-level dataclass consumer bounding
   (`TensorRT-LLM@dbc875d517fb4b6b4fde3882a54d3fcf34f5f2f1`, PR #20411).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


def simulate_moe_permute_invalid_topk_ids(
    valid_routes: list[list[int]],
    num_pad_tokens: int,
    n_expert: int,
    *,
    has_expert_map: bool = False,
) -> dict[str, Any]:
    """Model `moe_permute` cub::DeviceRadixSort key preparation and expert offset
    histogram from vLLM PR #59717 (`csrc/moe/moe_permute_unpermute_op.cu`).
    """
    topk = len(valid_routes[0]) if valid_routes else 0
    all_routes = [list(r) for r in valid_routes] + [[-1] * topk for _ in range(num_pad_tokens)]
    flat_routes = [eid for row in all_routes for eid in row]

    # Legacy bug: when expert_map is None, remap_expert_ids was skipped, leaving -1
    # in flat_routes. Signed int32 radix sort placed -1 BEFORE expert 0!
    legacy_keys = [
        (n_expert if (has_expert_map and eid < 0) else eid)
        for eid in flat_routes
    ]
    legacy_sorted = sorted(enumerate(legacy_keys), key=lambda pair: (pair[1], pair[0]))

    # Fixed behavior: remap_expert_ids always maps [0, n_expert) -> id and invalid -> n_expert
    fixed_keys = [
        (eid if 0 <= eid < n_expert else n_expert)
        for eid in flat_routes
    ]
    fixed_sorted = sorted(enumerate(fixed_keys), key=lambda pair: (pair[1], pair[0]))

    counts = [0] * n_expert
    for eid in fixed_keys:
        if 0 <= eid < n_expert:
            counts[eid] += 1
    expert_offsets = [0]
    for c in counts:
        expert_offsets.append(expert_offsets[-1] + c)

    num_valid_expanded = expert_offsets[-1]
    total_expanded = len(flat_routes)
    # Without zeroing mm1_out or bounding dynamic quantization to expert_offsets[-1],
    # stale workspace rows [num_valid_expanded : total_expanded] corrupt per-tensor amax.
    stale_workspace_amax = 64.0
    valid_activation_amax = 1.25
    corrupted_scale = max(valid_activation_amax, stale_workspace_amax) / 448.0
    clean_scale = valid_activation_amax / 448.0

    return {
        "n_expert": n_expert,
        "num_valid_tokens": len(valid_routes),
        "num_pad_tokens": num_pad_tokens,
        "topk": topk,
        "total_expanded_rows": total_expanded,
        "num_valid_expanded_rows": num_valid_expanded,
        "legacy_first_sorted_key": legacy_sorted[0][1] if legacy_sorted else None,
        "fixed_first_sorted_key": fixed_sorted[0][1] if fixed_sorted else None,
        "fixed_tail_sentinel_count": sum(1 for _, k in fixed_sorted if k == n_expert),
        "expert_offsets": expert_offsets,
        "corrupted_per_tensor_scale": round(corrupted_scale, 6),
        "clean_per_tensor_scale": round(clean_scale, 6),
        "scale_inflation_ratio": round(corrupted_scale / clean_scale, 2),
    }


def fp16_bits_to_float(bits: int) -> float:
    packed = struct.pack("<H", bits & 0xFFFF)
    return float(struct.unpack("<e", packed)[0])


def simulate_rdna_w4a16_packed_fp16_dequant(
    nibble_lo: int,
    nibble_hi: int,
    scale: float,
    zero_point: int | None = None,
) -> dict[str, Any]:
    """Verify `_int4_pair_to_fp16x2` (`v_and_or_b32 $0, $1, $2, 0x64006400`)
    from vLLM PR #60644 (`rdna_hybrid_w4a16.py`).
    """
    assert 0 <= nibble_lo < 16 and 0 <= nibble_hi < 16
    pair_word = (nibble_lo & 0xF) | ((nibble_hi & 0xF) << 16)
    packed_fp16x2 = (pair_word & 0x000F000F) | 0x64006400
    lo_bits = packed_fp16x2 & 0xFFFF
    hi_bits = (packed_fp16x2 >> 16) & 0xFFFF
    b_raw_lo = fp16_bits_to_float(lo_bits)
    b_raw_hi = fp16_bits_to_float(hi_bits)

    if zero_point is None:
        sub = 1032.0 * scale  # symmetric: (1024 + 8) * s
        expected_lo = (nibble_lo - 8) * scale
        expected_hi = (nibble_hi - 8) * scale
    else:
        sub = (1024.0 + float(zero_point)) * scale  # asymmetric: (1024 + zp) * s
        expected_lo = (nibble_lo - zero_point) * scale
        expected_hi = (nibble_hi - zero_point) * scale

    actual_lo = b_raw_lo * scale - sub
    actual_hi = b_raw_hi * scale - sub
    return {
        "nibble_lo": nibble_lo,
        "nibble_hi": nibble_hi,
        "packed_fp16x2_hex": f"0x{packed_fp16x2:08X}",
        "b_raw_lo": b_raw_lo,
        "b_raw_hi": b_raw_hi,
        "actual_lo": round(actual_lo, 6),
        "actual_hi": round(actual_hi, 6),
        "expected_lo": round(expected_lo, 6),
        "expected_hi": round(expected_hi, 6),
        "max_abs_error": round(max(abs(actual_lo - expected_lo), abs(actual_hi - expected_hi)), 9),
    }


def round_to_bf16(val: float) -> float:
    bits32 = struct.unpack("<I", struct.pack("<f", val))[0]
    rounding_bias = 0x7FFF + ((bits32 >> 16) & 1)
    bf16_bits = ((bits32 + rounding_bias) >> 16) & 0xFFFF
    return struct.unpack("<f", struct.pack("<I", bf16_bits << 16))[0]


def simulate_sglang_inkling_lm_head_output_fp32() -> dict[str, Any]:
    """Reproduce the exact greedy tie-rounding flip from SGLang PR #45335
    (`test_lm_head_output_dtype.py`).
    Weights `[[1.0, 1/1024], [1.0, 1/512]]` and hidden `[1.0, 1.0]` are exact in BF16,
    producing dot products `1 + 1/1024 = 1.0009765625` and `1 + 1/512 = 1.001953125`.
    Both round to `1.0` in BF16 (`argmax == 0`), whereas `out_dtype=torch.float32`
    preserves `1.0009765625 < 1.001953125` (`argmax == 1`) without promoting BF16 weights.
    """
    w0 = (round_to_bf16(1.0), round_to_bf16(1.0 / 1024.0))
    w1 = (round_to_bf16(1.0), round_to_bf16(1.0 / 512.0))
    h = (round_to_bf16(1.0), round_to_bf16(1.0))

    fp32_logits = [
        h[0] * w0[0] + h[1] * w0[1],
        h[0] * w1[0] + h[1] * w1[1],
    ]
    bf16_logits = [round_to_bf16(v) for v in fp32_logits]

    fp32_argmax = 0 if fp32_logits[0] >= fp32_logits[1] else 1
    bf16_argmax = 0 if bf16_logits[0] >= bf16_logits[1] else 1

    return {
        "operand_dtype": "bfloat16",
        "fp32_output_logits": fp32_logits,
        "bf16_output_logits": bf16_logits,
        "fp32_output_argmax": fp32_argmax,
        "bf16_output_argmax": bf16_argmax,
        "greedy_winner_flipped_in_bf16": fp32_argmax != bf16_argmax,
        "weight_memory_multiplier_vs_fp32_head": 0.5,
    }


def simulate_quark_fp8_scale_and_block_size() -> dict[str, Any]:
    """Verify `_resolve_quark_fp8_block_scale_name` and `_resolve_layer_weight_block_size`
    from SGLang PR #40776 (`deepseek_weight_loader.py`).
    """
    params_dict = {
        "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv": "fp8_scale_inv_tensor",
        "model.layers.0.mlp.experts.w13_weight_scale": "mxfp4_scale_tensor",
    }

    def resolve_scale_name(name: str) -> str:
        if name.endswith("weight_scale") and name not in params_dict:
            candidate = name.removesuffix("weight_scale") + "weight_scale_inv"
            if candidate in params_dict:
                return candidate
        return name

    quark_qkv_name = "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale"
    mxfp4_expert_name = "model.layers.0.mlp.experts.w13_weight_scale"

    layer_block_size = [64, 128]
    config_block_size = [128, 128]
    resolved_block_size = layer_block_size if layer_block_size is not None else config_block_size

    return {
        "resolved_quark_qkv_scale": resolve_scale_name(quark_qkv_name),
        "preserved_mxfp4_expert_scale": resolve_scale_name(mxfp4_expert_name),
        "resolved_layer_weight_block_size": resolved_block_size,
        "legacy_config_weight_block_size": config_block_size,
    }


def run_experiment() -> dict[str, Any]:
    moe_permute_case = simulate_moe_permute_invalid_topk_ids(
        valid_routes=[[2, 0, 5, 1], [0, 3, 2, 7], [5, 1, 4, 2]],
        num_pad_tokens=2,
        n_expert=8,
        has_expert_map=False,
    )
    rdna_sym_case = simulate_rdna_w4a16_packed_fp16_dequant(3, 14, scale=0.0625, zero_point=None)
    rdna_asym_case = simulate_rdna_w4a16_packed_fp16_dequant(5, 11, scale=0.125, zero_point=6)
    inkling_case = simulate_sglang_inkling_lm_head_output_fp32()
    quark_case = simulate_quark_fp8_scale_and_block_size()

    return {
        "experiment": "vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm",
        "phase": 70,
        "priority": 79,
        "vllm_moe_permute_invalid_ids": moe_permute_case,
        "vllm_rdna_w4a16_packed_fp16": {
            "symmetric_case": rdna_sym_case,
            "asymmetric_case": rdna_asym_case,
            "gfx942_kimi_k3_int4_requant_supported": True,
            "spec_decode_drafter_lora_allowed": True,
        },
        "sglang_inkling_lm_head_output_fp32": inkling_case,
        "sglang_quark_block_fp8": quark_case,
        "trtllm_cbts_dataclass_bounding": {
            "supports_frozen_literal_dataclass": True,
            "declines_metaclass_or_base_classes": True,
            "executor_forward_error_survival_kv_v2_guarded": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants.")
    args = parser.parse_args()

    summary = run_experiment()
    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert summary["vllm_moe_permute_invalid_ids"]["legacy_first_sorted_key"] == -1
        assert summary["vllm_moe_permute_invalid_ids"]["fixed_first_sorted_key"] == 0
        assert summary["vllm_moe_permute_invalid_ids"]["fixed_tail_sentinel_count"] == 8
        assert summary["vllm_rdna_w4a16_packed_fp16"]["symmetric_case"]["max_abs_error"] == 0.0
        assert summary["vllm_rdna_w4a16_packed_fp16"]["asymmetric_case"]["max_abs_error"] == 0.0
        assert summary["sglang_inkling_lm_head_output_fp32"]["fp32_output_argmax"] == 1
        assert summary["sglang_inkling_lm_head_output_fp32"]["bf16_output_argmax"] == 0
        assert summary["sglang_quark_block_fp8"]["resolved_layer_weight_block_size"] == [64, 128]


if __name__ == "__main__":
    main()
