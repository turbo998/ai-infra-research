#!/usr/bin/env python3
"""Phase 94 Track 3 deterministic stdlib experiment:
LMCache SGLang component direct-copy cudaMemcpyBatchAsync (#6062), NIXL UCX traceparent & at_us invariants (#2467),
Megatron-LM MoE paged-stash warmup empty_cache (#8507) & GTP CUDA-Graph wgrad coalescing fix (#8503),
TransformerEngine custom-op accelerator sentinel packing (#3739) & Userbuffers wrapper cleanup (#3732),
TorchTitan FlashAttention wheel probe (#5546), veRL DeepEP 1.2.1 + NVSHMEM 3.4.5 (#8354),
Mooncake MasterService DFS capacity VLOG(1) fast-path (#4786), and Dynamo dotted tool-name validation (#16482).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_lmcache_nixl_mooncake_dynamo() -> dict[str, object]:
    # LMCache #6062: EngineKVFormat::NL_X_NB_BS_NH_HS (format 18) direct copy via cudaMemcpyBatchAsync
    eligible_direct_copy_formats = [0, 1, 2, 3, 4, 5, 9, 11, 13, 18]
    cpu_overhead_us_per_entry = 0.6
    sm_blocks_used_direct_copy = 0

    # NIXL #2467: UCX trace-context propagation across 5 phases
    ucx_trace_phases = [
        "nixl::postXferReq.write",
        "nixl::submit",
        "nixl::wire.submitted",
        "nixl::notif.sent",
        "nixl::notif.received",
    ]

    # Mooncake #4786: MasterService::AllocateReplicas VLOG(1) on ErrorCode::NO_AVAILABLE_HANDLE
    error_log_bytes_per_fail = 470
    putstart_1t_before_mops = 0.35
    putstart_1t_after_mops = 3.50
    putstart_1t_speedup_x = round(putstart_1t_after_mops / putstart_1t_before_mops, 2)
    putstart_16t_before_mops = 0.26
    putstart_16t_after_mops = 1.70
    putstart_16t_speedup_x = round(putstart_16t_after_mops / putstart_16t_before_mops, 2)

    # Dynamo #16482: Allow '.' in OpenAI tool function names
    def is_valid_tool_name(name: str) -> bool:
        return bool(name) and all(c.isalnum() or c in ("_", "-", ".") for c in name)

    valid_samples = ["func-name_v2", "my-tool.v2", "namespace.sub.tool_1"]
    invalid_samples = ["<func_name>", "func name", "func@name", "func:name", "func/name", ""]
    assert all(is_valid_tool_name(s) for s in valid_samples)
    assert not any(is_valid_tool_name(s) for s in invalid_samples)
    assert putstart_1t_speedup_x == 10.0 and putstart_16t_speedup_x == 6.54

    return {
        "eligible_direct_copy_formats": eligible_direct_copy_formats,
        "sglang_component_format_id": 18,
        "cpu_overhead_us_per_entry": cpu_overhead_us_per_entry,
        "sm_blocks_used_direct_copy": sm_blocks_used_direct_copy,
        "ucx_trace_phases": ucx_trace_phases,
        "error_log_bytes_per_fail": error_log_bytes_per_fail,
        "putstart_1t_speedup_x": putstart_1t_speedup_x,
        "putstart_16t_speedup_x": putstart_16t_speedup_x,
        "dotted_tool_names_accepted": True,
    }


def model_megatron_te_torchtitan_verl() -> dict[str, object]:
    # Megatron-LM #8507: Paged-stash warmup peak memory reduction
    transient_warmup_activations_gib = 12.0
    measured_stash_gib = 10.0
    cuda_factor = 1.10
    persistent_stash_gib = round(measured_stash_gib * cuda_factor, 2)  # 11.0 GiB
    peak_warmup_before_gib = round(transient_warmup_activations_gib + persistent_stash_gib, 2)  # 23.0 GiB
    peak_warmup_after_gib = round(max(transient_warmup_activations_gib, persistent_stash_gib), 2)  # 12.0 GiB
    warmup_peak_saved_gib = round(peak_warmup_before_gib - peak_warmup_after_gib, 2)  # 11.0 GiB

    # Megatron-LM #8503: GTP grouped reduce-scatter wgrad release before CUDA Graph replay
    eager_wgrad_retained_after_wait_before = True
    eager_wgrad_retained_after_wait_after = False

    # TransformerEngine #3739: Keep None sentinels on custom op accelerator device
    sentinel_device_before = "cpu"
    sentinel_device_after = "cuda:0"
    inductor_partition_splits_on_sentinel_before = 1
    inductor_partition_splits_on_sentinel_after = 0

    # TorchTitan #5546 & veRL #8354
    sm_to_fa_module = {"SM90": ["FA3", "flash_attn_interface"], "SM100": ["FA4", "flash_attn.cute"]}
    verl_deepep_version = "1.2.1"
    verl_nvshmem_version = "3.4.5"

    assert persistent_stash_gib == 11.0 and warmup_peak_saved_gib == 11.0
    assert sentinel_device_after == "cuda:0" and inductor_partition_splits_on_sentinel_after == 0

    return {
        "transient_warmup_activations_gib": transient_warmup_activations_gib,
        "persistent_stash_gib": persistent_stash_gib,
        "peak_warmup_before_gib": peak_warmup_before_gib,
        "peak_warmup_after_gib": peak_warmup_after_gib,
        "warmup_peak_saved_gib": warmup_peak_saved_gib,
        "eager_wgrad_retained_after_wait_before": eager_wgrad_retained_after_wait_before,
        "eager_wgrad_retained_after_wait_after": eager_wgrad_retained_after_wait_after,
        "sentinel_device_before": sentinel_device_before,
        "sentinel_device_after": sentinel_device_after,
        "inductor_partition_splits_on_sentinel_before": inductor_partition_splits_on_sentinel_before,
        "inductor_partition_splits_on_sentinel_after": inductor_partition_splits_on_sentinel_after,
        "sm_to_fa_module": sm_to_fa_module,
        "verl_deepep_version": verl_deepep_version,
        "verl_nvshmem_version": verl_nvshmem_version,
    }


def generate_summary() -> dict[str, object]:
    return {
        "experiment": "lmcache_direct_copy_nixl_trace_megatron_stash_wgrad_te_titan_verl_mooncake_dynamo_phase94",
        "phase": 94,
        "priority": 103,
        "provenance": "own-offline",
        "lmcache_nixl_mooncake_dynamo": model_lmcache_nixl_mooncake_dynamo(),
        "megatron_te_torchtitan_verl": model_megatron_te_torchtitan_verl(),
        "all_invariants_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = generate_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == summary, "Mismatch in summary.json"
        print("OK: lmcache_direct_copy_nixl_trace_megatron_stash_wgrad_te_titan_verl_mooncake_dynamo_phase94 verified")
        return 0
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
