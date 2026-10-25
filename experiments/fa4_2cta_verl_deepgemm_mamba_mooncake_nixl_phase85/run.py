from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_summary() -> dict[str, Any]:
    # 1. FlashAttention-4 SM100/SM103 2-CTA dQaccum, hd128 reg cap & hd256 unification
    fa4_tile_m = 128
    fa4_tile_hdim = 128
    fa4_dqaccum_fp32_smem_bytes_saved = fa4_tile_m * fa4_tile_hdim * 4  # 65,536 B
    fa4_sdq_bf16_smem_bytes = fa4_tile_m * fa4_tile_hdim * 2  # 32,768 B
    fa4_sm103_q_stage = 1
    fa4_sm103_regs_softmax = 176
    fa4_sm103_regs_correction = 64
    fa4_sm103_threads_per_cta = 512
    fa4_sm103_launch_reg_cap = 65536 // fa4_sm103_threads_per_cta  # 128
    fa4_sm103_uncapped_regs_other = (
        (512 - fa4_sm103_regs_softmax * fa4_sm103_q_stage - fa4_sm103_regs_correction)
        // (3 - fa4_sm103_q_stage)
        // 8
        * 8
    )  # 136 -> illegal instruction in setmaxnreg.dec
    fa4_sm103_capped_regs_other = min(
        fa4_sm103_uncapped_regs_other, fa4_sm103_launch_reg_cap
    )  # 128
    fa4_hd256_removed_standalone_lines = 2007

    # 2. veRL unified QAT & DeepGEMM rollout on Blackwell (Qwen3.5-35B-A3B, EP=8, TP=2)
    verl_mxfp8_no_qat_k3_kl = 0.00387
    verl_mxfp8_with_qat_k3_kl = 0.00325
    verl_mxfp8_kl_reduction_ratio = round(
        verl_mxfp8_no_qat_k3_kl / verl_mxfp8_with_qat_k3_kl, 3
    )

    verl_mxfp4_no_qat_k3_kl = 0.00790
    verl_mxfp4_with_qat_k3_kl = 0.00140
    verl_mxfp4_kl_reduction_ratio = round(
        verl_mxfp4_no_qat_k3_kl / verl_mxfp4_with_qat_k3_kl, 3
    )

    verl_nvfp4_no_qat_k3_kl = 0.00808
    verl_nvfp4_with_qat_k3_kl = 0.00722
    verl_nvfp4_kl_reduction_ratio = round(
        verl_nvfp4_no_qat_k3_kl / verl_nvfp4_with_qat_k3_kl, 3
    )

    # 3. Megatron-Core Mamba ResidualBranchInput & dynamic dp_cp timeout
    megatron_default_nccl_timeout_sec = 600
    megatron_configured_timeout_sec = 45 * 60  # 2700 s
    megatron_dp_cp_world_size = 8
    megatron_valid_dynamic_cp_group_sizes = [1, 2, 4, 8]

    # 4. Mooncake Store standalone soft-pin & standby restore
    mooncake_soft_pin_max_batch_keys = 4096
    mooncake_default_soft_pin_ttl_ms = 1800000  # 30 min
    mooncake_prev_max_soft_pin_ttl_ms = 86400000  # 24 h
    mooncake_new_max_soft_pin_ttl_ms = 604800000  # 7 days
    mooncake_ttl_increase_factor = (
        mooncake_new_max_soft_pin_ttl_ms // mooncake_prev_max_soft_pin_ttl_ms
    )  # 7x

    # 5. NIXL Trace Plugin API V2, Dynamo /health & TE JAX MoE
    nixl_prev_plugin_api_version = 1
    nixl_new_plugin_api_version = 2
    dynamo_health_default_lists_instances = False
    dynamo_health_query_param = "instances=true"
    te_jax_split_dispatch_scan_mode_env = "NCCL_EP_HT_EM_AG_SCAN_MODE=1"

    return {
        "experiment_id": "fa4_2cta_verl_deepgemm_mamba_mooncake_nixl_phase85",
        "phase": 85,
        "priority": 94,
        "fa4_sm100_sm103": {
            "dqaccum_fp32_smem_bytes_saved": fa4_dqaccum_fp32_smem_bytes_saved,
            "sdq_bf16_smem_bytes": fa4_sdq_bf16_smem_bytes,
            "sm103_uncapped_regs_other": fa4_sm103_uncapped_regs_other,
            "sm103_launch_reg_cap": fa4_sm103_launch_reg_cap,
            "sm103_capped_regs_other": fa4_sm103_capped_regs_other,
            "hd256_removed_standalone_lines": fa4_hd256_removed_standalone_lines,
        },
        "verl_deepgemm_qat": {
            "mxfp8_no_qat_k3_kl": verl_mxfp8_no_qat_k3_kl,
            "mxfp8_with_qat_k3_kl": verl_mxfp8_with_qat_k3_kl,
            "mxfp8_kl_reduction_ratio": verl_mxfp8_kl_reduction_ratio,
            "mxfp4_no_qat_k3_kl": verl_mxfp4_no_qat_k3_kl,
            "mxfp4_with_qat_k3_kl": verl_mxfp4_with_qat_k3_kl,
            "mxfp4_kl_reduction_ratio": verl_mxfp4_kl_reduction_ratio,
            "nvfp4_no_qat_k3_kl": verl_nvfp4_no_qat_k3_kl,
            "nvfp4_with_qat_k3_kl": verl_nvfp4_with_qat_k3_kl,
            "nvfp4_kl_reduction_ratio": verl_nvfp4_kl_reduction_ratio,
        },
        "megatron_mamba_and_dp_cp": {
            "default_nccl_timeout_sec": megatron_default_nccl_timeout_sec,
            "configured_timeout_sec": megatron_configured_timeout_sec,
            "dp_cp_world_size": megatron_dp_cp_world_size,
            "valid_dynamic_cp_group_sizes": megatron_valid_dynamic_cp_group_sizes,
        },
        "mooncake_soft_pin": {
            "soft_pin_max_batch_keys": mooncake_soft_pin_max_batch_keys,
            "default_soft_pin_ttl_ms": mooncake_default_soft_pin_ttl_ms,
            "prev_max_soft_pin_ttl_ms": mooncake_prev_max_soft_pin_ttl_ms,
            "new_max_soft_pin_ttl_ms": mooncake_new_max_soft_pin_ttl_ms,
            "ttl_increase_factor": mooncake_ttl_increase_factor,
        },
        "nixl_dynamo_te": {
            "nixl_prev_plugin_api_version": nixl_prev_plugin_api_version,
            "nixl_new_plugin_api_version": nixl_new_plugin_api_version,
            "dynamo_health_default_lists_instances": dynamo_health_default_lists_instances,
            "dynamo_health_query_param": dynamo_health_query_param,
            "te_jax_split_dispatch_scan_mode_env": te_jax_split_dispatch_scan_mode_env,
        },
    }


def validate_summary(summary: dict[str, Any]) -> None:
    fa4 = summary["fa4_sm100_sm103"]
    assert fa4["dqaccum_fp32_smem_bytes_saved"] == 65536
    assert fa4["sm103_uncapped_regs_other"] == 136
    assert fa4["sm103_capped_regs_other"] == 128
    assert fa4["hd256_removed_standalone_lines"] == 2007

    verl = summary["verl_deepgemm_qat"]
    assert verl["mxfp4_kl_reduction_ratio"] == 5.643
    assert verl["mxfp8_kl_reduction_ratio"] == 1.191
    assert verl["nvfp4_kl_reduction_ratio"] == 1.119

    megatron = summary["megatron_mamba_and_dp_cp"]
    assert megatron["configured_timeout_sec"] == 2700

    mc = summary["mooncake_soft_pin"]
    assert mc["soft_pin_max_batch_keys"] == 4096
    assert mc["new_max_soft_pin_ttl_ms"] == 604800000
    assert mc["ttl_increase_factor"] == 7

    ndt = summary["nixl_dynamo_te"]
    assert ndt["nixl_new_plugin_api_version"] == 2
    assert ndt["dynamo_health_default_lists_instances"] is False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    validate_summary(summary)

    out_path = Path(__file__).with_name("summary.json")
    if args.check and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == summary, "summary.json drift detected"
    else:
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
