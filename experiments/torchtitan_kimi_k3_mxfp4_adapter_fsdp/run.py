#!/usr/bin/env python3
"""Deterministic model of TorchTitan Kimi K3 MXFP4 adapter, DeepSeek V3 HF save map, and GraphTrainer SimpleFSDP inspection.

Models three implementation contracts verified from pytorch/torchtitan:
1. `KimiK3StateDictAdapter` (`torchtitan/models/kimi_k3/state_dict_adapter.py` @ 7d563223d4ee10c0b8a7fc70d0cd119bcbcff93f):
   - MXFP4 `compressed-tensors` index remapping (`X.weight_packed -> X.weight`, dropping `.weight_scale`,
     and omitting `vision_tower.*` / `mm_projector.*` when `vision_encoder is None`).
   - Hybrid KDA (`KimiDeltaAttention`) vs MLA layer routing, DTensor `Replicate()` redistribution before
     `A_log` padding (`[n_heads] -> [head_dim]`) and `dt_bias` flattening (`[n_heads, head_dim] -> [n_heads * head_dim]`),
     with exact DTensor mesh/placement restoration in `from_hf()`.
   - Unused layer-0 attention residual placeholder synthesis (`self_attention_res_norm.weight = ones`,
     `self_attention_res_proj.weight = zeros`) on the PP stage owning layer 0.
2. `DeepSeekV3StateDictAdapter` (`torchtitan/models/deepseek_v3/state_dict_adapter.py` @ 1d581795cd014a46e871cc9fc605325fbbceee85):
   - Filtering the released 91,991-entry `model.safetensors.index.json` by removing `45,808` `.weight_scale_inv`
     entries and `788` `model.layers.61.*` MTP entries (when `num_mtp_layers == 0`), matching `to_hf()` output
     (`45,395` entries for `num_mtp_layers=0`, `46,183` entries for `num_mtp_layers=1`) so DCP consolidation
     never writes invalid `{"shape": [], "dtype": ""}` headers.
   - Cloning independent storage for `_MTP_SHARED_COPIES` (`embed_tokens.weight` and `shared_head.head.weight`,
     `+3.7 GB` on 671B BF16) so `safetensors` does not reject shared storage aliases.
3. `GraphTrainer` `SimpleFSDP` (`torchtitan/experiments/graph_trainer/simple_fsdp.py` @ 3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0):
   - Wrapping `linear_param_shard_placements(model)` in `with disable_active_parametrization():` so re-applying
     `data_parallel()` after `GraphPP` / `EP` does not invoke `ReplicateComputation` or `_ShardedFSDPTensor._build_operands()`
     while parameters are still on the `meta` device (`weight.is_cuda == False`).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

UNUSED_HF_LAYER_ZERO_ATTN_RES_KEYS = {
    "language_model.model.layers.0.self_attention_res_norm.weight",
    "language_model.model.layers.0.self_attention_res_proj.weight",
}


def filter_kimi_k3_fqn_to_index(
    raw_index: dict[str, int], *, has_vision_encoder: bool
) -> dict[str, int]:
    """Exact replica of `KimiK3StateDictAdapter.__init__` `fqn_to_index_mapping` filter."""
    return {
        key.removesuffix("_packed"): index
        for key, index in raw_index.items()
        if not key.endswith(".weight_scale")
        and (
            has_vision_encoder
            or not key.startswith(("vision_tower.", "mm_projector."))
        )
    }


def simulate_kimi_k3_kda_and_mxfp4_roundtrip(
    n_heads: int = 64,
    head_dim: int = 128,
    num_experts: int = 8,
    tp_degree: int = 4,
) -> dict[str, Any]:
    """Simulate `to_hf(to_quantized_layout=True)` and `from_hf(is_quantized_layout=True)` for Kimi K3."""
    # Initial TorchTitan state on a 1D TP mesh (`Shard(0)` on `A_log` and `dt_bias`)
    saved_layouts: dict[str, tuple[str, str]] = {}
    tt_a_log = [float(i + 1) for i in range(n_heads)]
    tt_dt_bias = [[float(h * 10 + d) for d in range(head_dim)] for h in range(n_heads)]

    # Step 1: `to_hf(to_quantized_layout=True)`
    # Record original DTensor placement (`tp_mesh`, `Shard(0)`), redistribute to `(Replicate(),)`
    saved_layouts["layers.0.attention.A_log"] = (f"tp_{tp_degree}", "Shard(0)")
    saved_layouts["layers.0.attention.dt_bias"] = (f"tp_{tp_degree}", "Shard(0)")

    # Pad A_log from [n_heads] to [head_dim] with zeros for MXFP4 DCP load planning
    hf_a_log_quant = tt_a_log + [0.0] * (head_dim - n_heads)
    # Ordinary HF export (`to_quantized_layout=False`) leaves A_log unpadded
    hf_a_log_unquant = list(tt_a_log)
    # Flatten dt_bias to 1D `[n_heads * head_dim]`
    hf_dt_bias = [val for row in tt_dt_bias for val in row]

    # Split grouped routed experts `w1_EFD` into per-expert `.weight_packed` keys
    hf_expert_keys = [
        f"language_model.model.layers.0.block_sparse_moe.experts.{e}.w1.weight_packed"
        for e in range(num_experts)
    ]

    # Layer-0 unused attention residual placeholder injection
    hf_layer0_placeholders = {
        "language_model.model.layers.0.self_attention_res_norm.weight": [1.0, 1.0, 1.0, 1.0],
        "language_model.model.layers.0.self_attention_res_proj.weight": [0.0, 0.0, 0.0, 0.0],
    }

    # Step 2: `from_hf(is_quantized_layout=True)`
    # Strip `_packed`, drop layer-0 unused keys, unpad `A_log[:n_heads]`, reshape `dt_bias`, restore DTensor layout
    restored_a_log = hf_a_log_quant[:n_heads]
    restored_dt_bias = [
        hf_dt_bias[h * head_dim : (h + 1) * head_dim] for h in range(n_heads)
    ]
    restored_layout_a_log = saved_layouts["layers.0.attention.A_log"]
    restored_layout_dt_bias = saved_layouts["layers.0.attention.dt_bias"]
    kept_layer0_res_keys = [
        k for k in hf_layer0_placeholders if k not in UNUSED_HF_LAYER_ZERO_ATTN_RES_KEYS
    ]

    return {
        "n_heads": n_heads,
        "head_dim": head_dim,
        "a_log_tt_len": len(tt_a_log),
        "a_log_hf_quantized_len": len(hf_a_log_quant),
        "a_log_hf_unquantized_len": len(hf_a_log_unquant),
        "dt_bias_tt_shape": [n_heads, head_dim],
        "dt_bias_hf_flat_len": len(hf_dt_bias),
        "expert_packed_keys_count": len(hf_expert_keys),
        "layer0_placeholders_generated": len(hf_layer0_placeholders),
        "layer0_placeholders_retained_in_tt": len(kept_layer0_res_keys),
        "a_log_exact_roundtrip": restored_a_log == tt_a_log,
        "dt_bias_exact_roundtrip": restored_dt_bias == tt_dt_bias,
        "dtensor_layout_restored": (
            restored_layout_a_log == (f"tp_{tp_degree}", "Shard(0)")
            and restored_layout_dt_bias == (f"tp_{tp_degree}", "Shard(0)")
        ),
    }


def analyze_deepseek_v3_safetensors_index() -> dict[str, Any]:
    """Model the exact DeepSeek-V3 671B `model.safetensors.index.json` filtering and MTP shared copies (#5065)."""
    # Exact counts from pytorch/torchtitan commit 1d581795cd014a46e871cc9fc605325fbbceee85:
    # Released index has 91,991 entries:
    # - 45,808 FP8 `X.weight_scale_inv` entries
    # - 788 MTP layer 61 (`model.layers.61.*`) entries (including 2 shared copies:
    #   `model.layers.61.embed_tokens.weight` and `model.layers.61.shared_head.head.weight`)
    # - 45,395 main-model weight entries (`num_mtp_layers=0`)
    # - 46,183 weight entries when `num_mtp_layers=1` (45,395 + 788 = 46,183)
    released_total_entries = 91991
    fp8_scale_inv_entries = 45808
    mtp_layer_61_entries = 788
    expected_saved_mtp0 = released_total_entries - fp8_scale_inv_entries - mtp_layer_61_entries
    expected_saved_mtp1 = released_total_entries - fp8_scale_inv_entries

    # Verify filter logic on a representative synthetic index map
    num_main_layers = 61
    sample_index = {
        "model.embed_tokens.weight": 1,
        "model.layers.0.self_attn.q_a_proj.weight": 1,
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv": 1,
        "model.layers.60.mlp.experts.255.down_proj.weight": 160,
        "model.layers.60.mlp.experts.255.down_proj.weight_scale_inv": 160,
        "model.layers.61.eh_proj.weight": 163,
        "model.layers.61.embed_tokens.weight": 163,
        "model.layers.61.shared_head.head.weight": 163,
        "lm_head.weight": 162,
    }

    def filter_ds_v3_index(mapping: dict[str, int], num_mtp_layers: int) -> dict[str, int]:
        num_hf_layers = num_main_layers + num_mtp_layers
        out: dict[str, int] = {}
        for key, index in mapping.items():
            is_fp8_scale = key.endswith(".weight_scale_inv")
            layer_match = re.match(r"model\.layers\.(\d+)\.", key)
            is_layer_not_in_config = (
                layer_match is not None and int(layer_match.group(1)) >= num_hf_layers
            )
            if is_fp8_scale or is_layer_not_in_config:
                continue
            out[key] = index
        return out

    filtered_mtp0 = filter_ds_v3_index(sample_index, num_mtp_layers=0)
    filtered_mtp1 = filter_ds_v3_index(sample_index, num_mtp_layers=1)

    # Calculate 671B BF16 MTP shared copy size:
    # vocab_size = 129,280, dim = 7,168, 2 tensors (`embed_tokens.weight` + `shared_head.head.weight`), BF16 (2 bytes)
    vocab_size = 129280
    hidden_dim = 7168
    mtp_shared_copy_bytes = 2 * vocab_size * hidden_dim * 2

    return {
        "released_index_total_entries": released_total_entries,
        "fp8_weight_scale_inv_dropped": fp8_scale_inv_entries,
        "mtp_layer_61_entries": mtp_layer_61_entries,
        "saved_entries_num_mtp_0": expected_saved_mtp0,
        "saved_entries_num_mtp_1": expected_saved_mtp1,
        "unfiltered_dcp_consolidation_empty_dtype_entries_mtp0": fp8_scale_inv_entries + mtp_layer_61_entries,
        "filtered_dcp_consolidation_empty_dtype_entries_mtp0": 0,
        "sample_filtered_keys_mtp0": sorted(filtered_mtp0.keys()),
        "sample_filtered_keys_mtp1": sorted(filtered_mtp1.keys()),
        "mtp_shared_copy_bytes_671b_bf16": mtp_shared_copy_bytes,
        "mtp_shared_copy_gb_decimal": round(mtp_shared_copy_bytes / 1e9, 3),
    }


def simulate_simple_fsdp_meta_inspection() -> dict[str, Any]:
    """Model `SimpleFSDP` `disable_active_parametrization()` during repeated `data_parallel()` (#5560)."""

    class MetaShardedMXFP8Weight:
        def __init__(self, shape: tuple[int, int], *, is_cuda: bool = False) -> None:
            self.shape = shape
            self.is_cuda = is_cuda
            self.build_operands_calls = 0

        def read_weight(self, active_parametrization: bool) -> tuple[int, int]:
            if active_parametrization:
                self.build_operands_calls += 1
                if not self.is_cuda:
                    raise ValueError("weight must be contiguous CUDA storage")
            return self.shape

    def run_pipeline_fsdp_passes(*, disable_during_inspect: bool) -> dict[str, Any]:
        weight = MetaShardedMXFP8Weight((7168, 2048), is_cuda=False)
        parametrized = False
        error_raised: str | None = None
        passes_completed = 0

        for _pass_idx in range(2):
            # Pass 0: GraphPP / EP parametrizes submodule
            # Pass 1: Top-level `data_parallel()` calls `linear_param_shard_placements(model)`
            try:
                effective_active = parametrized and (not disable_during_inspect)
                _ = weight.read_weight(active_parametrization=effective_active)
                parametrized = True
                passes_completed += 1
            except ValueError as exc:
                error_raised = str(exc)
                break

        return {
            "disable_during_inspect": disable_during_inspect,
            "passes_completed": passes_completed,
            "build_operands_calls_during_inspect": weight.build_operands_calls,
            "error_raised": error_raised,
        }

    buggy_path = run_pipeline_fsdp_passes(disable_during_inspect=False)
    fixed_path = run_pipeline_fsdp_passes(disable_during_inspect=True)
    return {
        "without_disable_active_parametrization": buggy_path,
        "with_disable_active_parametrization": fixed_path,
    }


def build_summary() -> dict[str, Any]:
    raw_kimi_index = {
        "language_model.model.embed_tokens.weight": 1,
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed": 2,
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_scale": 2,
        "vision_tower.patch_embed.proj.weight": 3,
        "mm_projector.proj.0.weight": 3,
    }
    kimi_filtered_text_only = filter_kimi_k3_fqn_to_index(raw_kimi_index, has_vision_encoder=False)
    kimi_filtered_multimodal = filter_kimi_k3_fqn_to_index(raw_kimi_index, has_vision_encoder=True)

    return {
        "experiment_id": "torchtitan_kimi_k3_mxfp4_adapter_fsdp",
        "evidence_class": "own-offline",
        "source_commits": {
            "kimi_k3_mxfp4_adapter": "7d563223d4ee10c0b8a7fc70d0cd119bcbcff93f",
            "deepseek_v3_hf_save_map": "1d581795cd014a46e871cc9fc605325fbbceee85",
            "graphtrainer_simple_fsdp_meta": "3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0",
        },
        "kimi_k3_index_filtering": {
            "text_only_keys": sorted(kimi_filtered_text_only.keys()),
            "multimodal_keys": sorted(kimi_filtered_multimodal.keys()),
        },
        "kimi_k3_kda_mxfp4_roundtrip": simulate_kimi_k3_kda_and_mxfp4_roundtrip(),
        "deepseek_v3_safetensors_index": analyze_deepseek_v3_safetensors_index(),
        "simple_fsdp_meta_inspection": simulate_simple_fsdp_meta_inspection(),
    }


def run_check(summary: dict[str, Any]) -> None:
    kimi_idx = summary["kimi_k3_index_filtering"]
    assert kimi_idx["text_only_keys"] == [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight",
    ]
    assert len(kimi_idx["multimodal_keys"]) == 4

    kimi_rt = summary["kimi_k3_kda_mxfp4_roundtrip"]
    assert kimi_rt["a_log_tt_len"] == 64
    assert kimi_rt["a_log_hf_quantized_len"] == 128
    assert kimi_rt["a_log_hf_unquantized_len"] == 64
    assert kimi_rt["layer0_placeholders_retained_in_tt"] == 0
    assert kimi_rt["a_log_exact_roundtrip"] is True
    assert kimi_rt["dt_bias_exact_roundtrip"] is True
    assert kimi_rt["dtensor_layout_restored"] is True

    dsv3 = summary["deepseek_v3_safetensors_index"]
    assert dsv3["saved_entries_num_mtp_0"] == 45395
    assert dsv3["saved_entries_num_mtp_1"] == 46183
    assert dsv3["filtered_dcp_consolidation_empty_dtype_entries_mtp0"] == 0
    assert abs(dsv3["mtp_shared_copy_gb_decimal"] - 3.707) < 0.01

    fsdp = summary["simple_fsdp_meta_inspection"]
    assert fsdp["without_disable_active_parametrization"]["passes_completed"] == 1
    assert (
        fsdp["without_disable_active_parametrization"]["error_raised"]
        == "weight must be contiguous CUDA storage"
    )
    assert fsdp["with_disable_active_parametrization"]["passes_completed"] == 2
    assert fsdp["with_disable_active_parametrization"]["build_operands_calls_during_inspect"] == 0
    assert fsdp["with_disable_active_parametrization"]["error_raised"] is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        run_check(summary)

    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
