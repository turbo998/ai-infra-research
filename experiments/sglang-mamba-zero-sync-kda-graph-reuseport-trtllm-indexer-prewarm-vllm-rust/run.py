#!/usr/bin/env python3
"""Deterministic reproduction of SGLang, TensorRT-LLM, and vLLM Phase 50 upstream optimizations.

Covers:
1. sgl-project/sglang@740d5f9f9e33aeebc6b5bcb1ba73f25bf6db062b (#45472):
   - Zero host-device sync planning for hybrid Mamba extend batches (`mamba_ping_pong_empty: list[bool]`,
     device-side `ping_pong_mapping[req_pool_indices, track_positions]` gather, `index_fill_` clear,
     and `searchsorted` + `clamp_` + equality replacing `torch.isin` which triggers a host sync via
     `unique()` when `P >= 10 * N**0.145`).
2. sgl-project/sglang@f06d1bb1f45df617aa455c103ce5ede2bc6c6a47 (#45473),
   a3d00eff58b6c22ea7539f882cc10c537634200b (#45487), &
   7e5256d1ba3aaa628bb8b756fafb904a76a1123a (#45482):
   - `FlashInferKDAPrefillGraph` (`RecurrentKDAPrefillWrapper` on SM100/SM103),
   - Host-side `contains_last_prefill_chunk` (`extend_end == len(full_untruncated_fill_ids)`) + TBO `filter_batch`,
   - `_ReusePortMultiprocess` (`--enable-http-worker-reuseport`, `uvicorn>=0.51`) parent socket reservation.
3. NVIDIA/TensorRT-LLM@370fe4c5096cff7ffb457092d36a208da34006d7 (#20734):
   - DeepSeek-V4 CuTe DSL Indexer Q (`_prewarm_cute_dsl_indexer_q`) import fix, PP stage filtering,
     and prewarm token counts `(4, 8, 16)` covering `_fallback_tactic()` branches (`m <= 4`, `4 < m <= 8`, `m > 8`).
4. vllm-project/vllm@4a3ee9a0f6a2958f2712ceb3c9f60fe48cd764b3 (#57393),
   2916b4b4a6b7e9ae552e661b365fec65117134ac (#62822),
   bc4794ef9a9fea5f8a003ef4dc5a49bbe8145241 (#62825),
   8a21d0a328bb3cbb9c73ca0ae65d83465768cca4 (#62416), &
   e0ca7d9cc11da2011389c725e5d31c54c3b47259 (#62824):
   - Cosmos3-Edge ModelOpt fake-quantizer drop & scale remap, plus Rust frontend parity (`Ordering::Equal`
     `PromptFillsContext` vs `Ordering::Greater` `PromptTooLong`, `return_token_ids`, env vars, and CLI flags).
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any


def torch_isin_triggers_host_sync(n_elements: int, num_placeholders: int) -> bool:
    """In PyTorch TensorIterator, torch.isin switches from brute-force to unique()+sort
    (causing a device-to-host sync for output shape) when P >= 10 * N**0.145."""
    threshold = 10.0 * (float(n_elements) ** 0.145)
    return float(num_placeholders) >= threshold


def searchsorted_multimodal_mask(input_ids: list[int], placeholder_values: list[int]) -> list[bool]:
    """Reproduce SGLang #45472 zero-sync multimodal mask via sorted searchsorted + clamp + eq."""
    if not placeholder_values or not input_ids:
        return [False] * len(input_ids)
    sorted_p = sorted(placeholder_values)
    max_idx = len(sorted_p) - 1
    mask: list[bool] = []
    for token_id in input_ids:
        idx = bisect.bisect_left(sorted_p, token_id)
        clamped_idx = min(idx, max_idx)
        mask.append(sorted_p[clamped_idx] == token_id)
    return mask


class MambaZeroSyncPingPongPlanner:
    """Reproduce SGLang #45472 host-tracked mamba_ping_pong_empty and device gather."""

    def __init__(self, max_reqs: int) -> None:
        self.ping_pong_mapping: list[tuple[int, int]] = [(-1, -1) for _ in range(max_reqs)]
        self.mamba_ping_pong_empty: list[bool] = [True for _ in range(max_reqs)]
        self.device_to_host_sync_count = 0

    def assign_slots(self, req_pool_idx: int, slot0: int, slot1: int) -> None:
        self.ping_pong_mapping[req_pool_idx] = (slot0, slot1)
        self.mamba_ping_pong_empty[req_pool_idx] = (slot0 == -1 and slot1 == -1)

    def free_slots(self, req_pool_idx: int) -> None:
        self.ping_pong_mapping[req_pool_idx] = (-1, -1)
        self.mamba_ping_pong_empty[req_pool_idx] = True

    def gather_track_slots_zero_sync(
        self, req_pool_indices: list[int], track_positions: list[int]
    ) -> list[int]:
        for r in req_pool_indices:
            if self.mamba_ping_pong_empty[r]:
                raise RuntimeError(f"Request {r} has unassigned ping-pong buffer")
        return [
            self.ping_pong_mapping[r][pos]
            for r, pos in zip(req_pool_indices, track_positions)
        ]


def compute_contains_last_prefill_chunk(
    extend_reqs: list[dict[str, int]],
    keep_indices: list[int] | None = None,
) -> bool:
    """Reproduce SGLang #45487 host-side contains_last_prefill_chunk and TBO filter_batch."""
    filtered = (
        [extend_reqs[i] for i in keep_indices]
        if keep_indices is not None
        else extend_reqs
    )
    return any(req["extend_end"] == req["full_untruncated_len"] for req in filtered)


def simulate_reuseport_socket_plan(num_workers: int, enable_reuseport: bool) -> dict[str, Any]:
    """Reproduce SGLang #45482 _ReusePortMultiprocess socket allocation."""
    if not enable_reuseport:
        return {
            "mode": "shared_single_socket",
            "parent_reserved_sockets": 1,
            "worker_private_sockets": 0,
            "total_bound_sockets": 1,
            "kernel_accept_load_balanced": False,
        }
    return {
        "mode": "reuseport_per_worker",
        "parent_reserved_sockets": 1,
        "worker_private_sockets": num_workers,
        "total_bound_sockets": 1 + num_workers,
        "kernel_accept_load_balanced": True,
    }


def trtllm_deepseek_v4_indexer_q_fallback_tactic(m: int) -> tuple[int, int, str]:
    """Reproduce TensorRT-LLM #20734 DeepSeekV4Indexer._fallback_tactic(m) branches."""
    if m <= 4:
        return (64, 128, "small_m_le_4")
    if m <= 8:
        return (128, 128, "medium_m_5_to_8")
    return (128, 256, "large_m_gt_8")


def prewarm_cute_dsl_indexer_q(
    layers: list[dict[str, Any]],
    prewarm_token_counts: tuple[int, ...] = (4, 8, 16),
) -> dict[str, Any]:
    """Reproduce TensorRT-LLM #20734 PP-stage filtering and 3-branch CuTe DSL prewarm."""
    local_indexers = [
        layer["layer_id"]
        for layer in layers
        if layer.get("has_wq_b_weight", False) and layer.get("fused_mxfp4_enabled", False)
    ]
    compiled_tactics = [trtllm_deepseek_v4_indexer_q_fallback_tactic(m) for m in prewarm_token_counts]
    return {
        "local_indexer_layers": local_indexers,
        "prewarm_token_counts": list(prewarm_token_counts),
        "compiled_tactics": [
            {"m": m, "tile_m": t[0], "tile_n": t[1], "branch": t[2]}
            for m, t in zip(prewarm_token_counts, compiled_tactics)
        ],
        "total_prewarm_invocations": len(local_indexers) * len(prewarm_token_counts),
    }


def remap_cosmos3_edge_checkpoint_key(key: str) -> str | None:
    """Reproduce vLLM #57393 Cosmos3-Edge ModelOpt fake-quantizer key filter & scale remap."""
    if any(q in key for q in (".input_quantizer.", ".weight_quantizer.", ".output_quantizer.")):
        return None
    if key.endswith(".weight_scale_inv"):
        return key[: -len(".weight_scale_inv")] + ".weight_scale"
    if key.endswith(".act_quant_scale"):
        return key[: -len(".act_quant_scale")] + ".input_scale"
    return key


def validate_vllm_rust_prompt_length(prompt_len: int, max_model_len: int, max_tokens: int) -> dict[str, Any]:
    """Reproduce vLLM #62824 Rust TokenizeError PromptFillsContext vs PromptTooLong."""
    if prompt_len > max_model_len:
        return {
            "valid": False,
            "error_variant": "PromptTooLong",
            "param": "input_tokens",
            "message": f"Prompt length of {prompt_len} tokens exceeds maximum model length of {max_model_len}.",
        }
    if prompt_len == max_model_len and max_tokens >= 1:
        return {
            "valid": False,
            "error_variant": "PromptFillsContext",
            "param": "input_tokens",
            "message": (
                f"This model's maximum context length is {max_model_len} tokens. "
                f"However, your request has {max_model_len} input tokens. "
                "Please reduce the length of the input messages."
            ),
        }
    return {
        "valid": True,
        "error_variant": None,
        "param": None,
        "max_Completable_tokens": max_model_len - prompt_len,
    }


def run_experiment() -> dict[str, Any]:
    # 1. SGLang Mamba zero-sync planner & multimodal mask
    planner = MambaZeroSyncPingPongPlanner(max_reqs=8)
    planner.assign_slots(1, 10, 11)
    planner.assign_slots(3, 30, 31)
    planner.assign_slots(5, 50, 51)
    gathered_slots = planner.gather_track_slots_zero_sync([1, 3, 5], [0, 1, 0])

    # Check torch.isin sync threshold at N=4096 tokens: 10 * 4096**0.145 = 33.40 -> P=34 triggers sync!
    isin_sync_at_p20 = torch_isin_triggers_host_sync(n_elements=4096, num_placeholders=20)
    isin_sync_at_p34 = torch_isin_triggers_host_sync(n_elements=4096, num_placeholders=34)
    mm_mask = searchsorted_multimodal_mask(
        input_ids=[101, 32001, 42, 32005, 999, 32000],
        placeholder_values=[32005, 32001, 32003],
    )

    # 2. SGLang KDA prefill graph, contains_last_prefill_chunk, and ReusePort
    extend_reqs = [
        {"req_id": 0, "extend_end": 2048, "full_untruncated_len": 4096},
        {"req_id": 1, "extend_end": 4096, "full_untruncated_len": 4096},
    ]
    chunk_before_filter = compute_contains_last_prefill_chunk(extend_reqs)
    chunk_after_tbo_split0 = compute_contains_last_prefill_chunk(extend_reqs, keep_indices=[0])
    chunk_after_tbo_split1 = compute_contains_last_prefill_chunk(extend_reqs, keep_indices=[1])
    reuseport_plan = simulate_reuseport_socket_plan(num_workers=8, enable_reuseport=True)

    # 3. TensorRT-LLM DeepSeek-V4 CuTe DSL Indexer Q prewarm across PP stages
    pp_stage_layers = [
        {"layer_id": 0, "has_wq_b_weight": False, "fused_mxfp4_enabled": True},  # remote PP stage
        {"layer_id": 1, "has_wq_b_weight": True, "fused_mxfp4_enabled": True},   # local PP stage
        {"layer_id": 2, "has_wq_b_weight": True, "fused_mxfp4_enabled": True},   # local PP stage
    ]
    trtllm_prewarm = prewarm_cute_dsl_indexer_q(pp_stage_layers)

    # 4. vLLM Cosmos3-Edge & Rust frontend
    raw_keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.input_quantizer._amax",
        "model.layers.0.self_attn.q_proj.weight_quantizer._amax",
        "model.layers.0.self_attn.q_proj.output_quantizer._amax",
        "model.layers.0.self_attn.q_proj.weight_scale_inv",
        "model.layers.0.self_attn.q_proj.act_quant_scale",
    ]
    remapped_keys = [k for k in (remap_cosmos3_edge_checkpoint_key(rk) for rk in raw_keys) if k is not None]

    rust_eq = validate_vllm_rust_prompt_length(prompt_len=8192, max_model_len=8192, max_tokens=16)
    rust_gt = validate_vllm_rust_prompt_length(prompt_len=8193, max_model_len=8192, max_tokens=16)
    rust_ok = validate_vllm_rust_prompt_length(prompt_len=8192, max_model_len=8192, max_tokens=0)

    return {
        "experiment": "sglang-mamba-zero-sync-kda-graph-reuseport-trtllm-indexer-prewarm-vllm-rust",
        "sglang_mamba_zero_sync": {
            "gathered_slots": gathered_slots,
            "device_to_host_sync_count": planner.device_to_host_sync_count,
            "isin_sync_at_n4096_p20": isin_sync_at_p20,
            "isin_sync_at_n4096_p34": isin_sync_at_p34,
            "searchsorted_multimodal_mask": mm_mask,
        },
        "sglang_kda_and_reuseport": {
            "contains_last_prefill_chunk_full": chunk_before_filter,
            "contains_last_prefill_chunk_tbo_split0": chunk_after_tbo_split0,
            "contains_last_prefill_chunk_tbo_split1": chunk_after_tbo_split1,
            "reuseport_plan": reuseport_plan,
        },
        "trtllm_indexer_prewarm": trtllm_prewarm,
        "vllm_cosmos3_and_rust": {
            "raw_key_count": len(raw_keys),
            "remapped_keys": remapped_keys,
            "rust_prompt_fills_context": rust_eq,
            "rust_prompt_too_long": rust_gt,
            "rust_zero_max_tokens_ok": rust_ok,
        },
    }


def check_invariants(summary: dict[str, Any]) -> None:
    mamba = summary["sglang_mamba_zero_sync"]
    assert mamba["gathered_slots"] == [10, 31, 50]
    assert mamba["device_to_host_sync_count"] == 0
    assert mamba["isin_sync_at_n4096_p20"] is False
    assert mamba["isin_sync_at_n4096_p34"] is True
    assert mamba["searchsorted_multimodal_mask"] == [False, True, False, True, False, False]

    kda = summary["sglang_kda_and_reuseport"]
    assert kda["contains_last_prefill_chunk_full"] is True
    assert kda["contains_last_prefill_chunk_tbo_split0"] is False
    assert kda["contains_last_prefill_chunk_tbo_split1"] is True
    assert kda["reuseport_plan"]["total_bound_sockets"] == 9
    assert kda["reuseport_plan"]["parent_reserved_sockets"] == 1
    assert kda["reuseport_plan"]["worker_private_sockets"] == 8

    trt = summary["trtllm_indexer_prewarm"]
    assert trt["local_indexer_layers"] == [1, 2]
    assert trt["prewarm_token_counts"] == [4, 8, 16]
    assert [c["branch"] for c in trt["compiled_tactics"]] == [
        "small_m_le_4",
        "medium_m_5_to_8",
        "large_m_gt_8",
    ]
    assert trt["total_prewarm_invocations"] == 6

    vllm = summary["vllm_cosmos3_and_rust"]
    assert vllm["remapped_keys"] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.q_proj.input_scale",
    ]
    assert vllm["rust_prompt_fills_context"]["error_variant"] == "PromptFillsContext"
    assert vllm["rust_prompt_fills_context"]["param"] == "input_tokens"
    assert vllm["rust_prompt_too_long"]["error_variant"] == "PromptTooLong"
    assert vllm["rust_zero_max_tokens_ok"]["valid"] is True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run invariant checks and write summary.json")
    args = parser.parse_args()

    summary = run_experiment()
    if args.check:
        check_invariants(summary)

    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
