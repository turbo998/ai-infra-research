from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_torchtitan_fsdp_inspection_guard() -> dict[str, Any]:
    def run_reapply(disable_parametrization: bool) -> dict[str, Any]:
        build_operands_calls = 0
        raised_meta_storage_error = False
        for pass_idx in (1, 2):
            already_parametrized = pass_idx == 2
            if already_parametrized and not disable_parametrization:
                build_operands_calls += 1
                raised_meta_storage_error = True
        return {
            "disable_active_parametrization": disable_parametrization,
            "build_operands_calls_on_meta": build_operands_calls,
            "raised_dist_moe_cuda_storage_error": raised_meta_storage_error,
            "inspected_raw_sharded_parameter": disable_parametrization,
        }

    return {
        "unguarded_inspection": run_reapply(False),
        "guarded_inspection": run_reapply(True),
    }


def simulate_kimi_k3_mxfp4_state_dict_adapter() -> dict[str, Any]:
    total_hf_names_full_model = 249924
    num_safetensors_files = 96

    sample_index = {
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed": "model-00001-of-00096.safetensors",
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_scale": "model-00001-of-00096.safetensors",
        "language_model.model.layers.0.self_attn.A_log": "model-00001-of-00096.safetensors",
        "language_model.model.layers.0.self_attn.dt_bias": "model-00001-of-00096.safetensors",
        "vision_tower.encoder.blocks.0.wqkv.weight": "model-00096-of-00096.safetensors",
        "mm_projector.linear.weight": "model-00096-of-00096.safetensors",
    }

    def remap_index(vision_enabled: bool) -> dict[str, str]:
        return {
            key.removesuffix("_packed"): shard
            for key, shard in sample_index.items()
            if not key.endswith(".weight_scale")
            and (
                vision_enabled
                or not key.startswith(("vision_tower.", "mm_projector."))
            )
        }

    n_heads = 16
    head_dim = 128
    a_log_pad_quantized = head_dim - n_heads
    dt_bias_flat_len = n_heads * head_dim

    return {
        "full_model_hf_key_count": total_hf_names_full_model,
        "released_safetensors_files": num_safetensors_files,
        "quantized_reader_default_threads": 4,
        "quantized_reader_target_dtype": "bfloat16",
        "remapped_index_multimodal_keys": sorted(remap_index(True).keys()),
        "remapped_index_text_only_keys": sorted(remap_index(False).keys()),
        "kda_layout_adaptation": {
            "n_heads": n_heads,
            "head_dim": head_dim,
            "a_log_titan_shape": [n_heads],
            "a_log_quantized_hf_shape": [head_dim],
            "a_log_unquantized_hf_shape": [n_heads],
            "a_log_pad_elements": a_log_pad_quantized,
            "dt_bias_titan_shape": [n_heads, head_dim],
            "dt_bias_hf_flat_shape": [dt_bias_flat_len],
            "dtensor_temporary_placement": "Replicate",
            "restores_original_mesh_placement_in_from_hf": True,
        },
    }


def simulate_megatron_idle_rerun_strict_load() -> dict[str, Any]:
    def evaluate_strict_load(
        saved_state: str, validate_returns: bool, with_idle_fix: bool
    ) -> dict[str, Any]:
        restore_rerun_state = validate_returns
        idle_rerun_state = saved_state == "NOT_RUNNING_YET"
        if with_idle_fix:
            include_in_gen_sd = restore_rerun_state or idle_rerun_state
            ignore_rerun_state = not restore_rerun_state
        else:
            include_in_gen_sd = restore_rerun_state
            ignore_rerun_state = not restore_rerun_state

        unexpected_keys = 0 if include_in_gen_sd else 1
        strict_raise_all_passes = unexpected_keys == 0
        rerun_resumed = not ignore_rerun_state
        return {
            "saved_state": saved_state,
            "validate_state_dict": validate_returns,
            "with_idle_fix": with_idle_fix,
            "included_in_load_schema": include_in_gen_sd,
            "ignore_rerun_state": ignore_rerun_state,
            "unexpected_rerun_keys": unexpected_keys,
            "strict_raise_all_passes": strict_raise_all_passes,
            "bogus_rerun_resumed": rerun_resumed and idle_rerun_state,
        }

    return {
        "legacy_idle_checkpoint_strict_load": evaluate_strict_load(
            "NOT_RUNNING_YET", False, False
        ),
        "fixed_idle_checkpoint_strict_load": evaluate_strict_load(
            "NOT_RUNNING_YET", False, True
        ),
        "active_rerun_checkpoint_strict_load": evaluate_strict_load(
            "RUNNING", True, True
        ),
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "torchtitan_fsdp_inspect_kimi_k3_mxfp4_megatron_rerun",
        "evidence_class": "own-offline",
        "torchtitan_fsdp_inspection": simulate_torchtitan_fsdp_inspection_guard(),
        "torchtitan_kimi_k3_mxfp4_adapter": simulate_kimi_k3_mxfp4_state_dict_adapter(),
        "megatron_idle_rerun_strict_load": simulate_megatron_idle_rerun_strict_load(),
    }


def verify_summary(summary: dict[str, Any]) -> None:
    fsdp = summary["torchtitan_fsdp_inspection"]
    assert fsdp["unguarded_inspection"]["raised_dist_moe_cuda_storage_error"]
    assert not fsdp["guarded_inspection"]["raised_dist_moe_cuda_storage_error"]
    assert fsdp["guarded_inspection"]["build_operands_calls_on_meta"] == 0

    kimi = summary["torchtitan_kimi_k3_mxfp4_adapter"]
    assert kimi["full_model_hf_key_count"] == 249924
    assert kimi["released_safetensors_files"] == 96
    assert len(kimi["remapped_index_multimodal_keys"]) == 5
    assert len(kimi["remapped_index_text_only_keys"]) == 3
    assert kimi["kda_layout_adaptation"]["a_log_pad_elements"] == 112
    assert kimi["kda_layout_adaptation"]["dt_bias_hf_flat_shape"] == [2048]

    meg = summary["megatron_idle_rerun_strict_load"]
    assert not meg["legacy_idle_checkpoint_strict_load"]["strict_raise_all_passes"]
    assert meg["fixed_idle_checkpoint_strict_load"]["strict_raise_all_passes"]
    assert meg["fixed_idle_checkpoint_strict_load"]["ignore_rerun_state"]
    assert not meg["fixed_idle_checkpoint_strict_load"]["bogus_rerun_resumed"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    summary = build_summary()
    verify_summary(summary)

    out_paths = [base_dir / "summary.json", base_dir / "results" / "summary.json"]
    if args.check:
        for p in out_paths:
            existing = json.loads(p.read_text(encoding="utf-8"))
            verify_summary(existing)
            assert existing == summary, f"Mismatch in {p}"
        print("torchtitan_fsdp_inspect_kimi_k3_mxfp4_megatron_rerun: check passed")
        return

    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rendered, encoding="utf-8")
    print("torchtitan_fsdp_inspect_kimi_k3_mxfp4_megatron_rerun: generated summary.json")


if __name__ == "__main__":
    main()
