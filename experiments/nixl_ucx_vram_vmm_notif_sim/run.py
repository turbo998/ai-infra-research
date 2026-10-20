#!/usr/bin/env python3
"""Deterministic simulator for NVIDIA NIXL Distributed KV Transfer Engine:
UCX vram_memtype_hint bitmask resolution, Libfabric cuMemCreate primary-context fallback,
and Python empty-notification (b"") consumption semantics.

Models the exact control-flow and bitmask invariants from ai-dynamo/nixl commits:
- 466ebb1ccc5a6e4e8cbaf954c94020400c9ae876 (src/plugins/ucx/ucx_utils.cpp, ucx_backend.cpp)
- 1f4afe6b60b2b5114deff13a4d2653e7ba683f07 (src/plugins/libfabric/libfabric_backend.cpp)
- b55e79152fd67a0b7214f9127e435ad1820b2f78 (src/api/python/_api.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass


UCS_MEMORY_TYPE_HOST = 0
UCS_MEMORY_TYPE_CUDA = 1
UCS_MEMORY_TYPE_CUDA_MANAGED = 2
UCS_MEMORY_TYPE_ROCM = 3
UCS_MEMORY_TYPE_ZE_DEVICE = 4
UCS_MEMORY_TYPE_UNKNOWN = 5

MEMTYPE_NAMES = {
    UCS_MEMORY_TYPE_HOST: "host",
    UCS_MEMORY_TYPE_CUDA: "cuda",
    UCS_MEMORY_TYPE_CUDA_MANAGED: "cuda-managed",
    UCS_MEMORY_TYPE_ROCM: "rocm",
    UCS_MEMORY_TYPE_ZE_DEVICE: "ze-device",
    UCS_MEMORY_TYPE_UNKNOWN: "unknown",
}


def ucs_bit(mem_type: int) -> int:
    return 1 << mem_type


def supports_memory_type(mask: int, mem_type: int) -> bool:
    return (mask & ucs_bit(mem_type)) != 0


def ucx_auto_detect_vram_hint(mask: int) -> int:
    """Exact replica of ucx_auto_detect_vram_hint in src/plugins/ucx/ucx_utils.cpp."""
    has_cuda = supports_memory_type(mask, UCS_MEMORY_TYPE_CUDA) or supports_memory_type(
        mask, UCS_MEMORY_TYPE_CUDA_MANAGED
    )
    has_rocm = supports_memory_type(mask, UCS_MEMORY_TYPE_ROCM)
    has_ze = supports_memory_type(mask, UCS_MEMORY_TYPE_ZE_DEVICE)

    if has_ze and not has_cuda and not has_rocm:
        return UCS_MEMORY_TYPE_ZE_DEVICE
    return UCS_MEMORY_TYPE_UNKNOWN


@dataclass(frozen=True)
class UcxVramRegResult:
    scenario: str
    policy: str
    supported_mask: int
    resolved_hint: str
    passed_hint_field_to_ucp: bool
    detected_mem_type: str
    registration_succeeded: bool
    unmapped_on_host_misdetection: bool


def simulate_ucx_vram_registration(
    scenario: str,
    policy: str,
    supported_mask: int,
    ucx_detected_without_hint: int,
) -> UcxVramRegResult:
    """Model nixlUcxContext::resolveMemoryTypeConfig and memRegInternal for VRAM_SEG."""
    policy_to_type = {
        "cuda": UCS_MEMORY_TYPE_CUDA,
        "cuda-managed": UCS_MEMORY_TYPE_CUDA_MANAGED,
        "rocm": UCS_MEMORY_TYPE_ROCM,
        "ze-device": UCS_MEMORY_TYPE_ZE_DEVICE,
    }
    if policy == "none":
        vram_hint = UCS_MEMORY_TYPE_UNKNOWN
    elif policy == "auto":
        vram_hint = ucx_auto_detect_vram_hint(supported_mask)
    elif policy in policy_to_type:
        requested = policy_to_type[policy]
        if not supports_memory_type(supported_mask, requested):
            raise ValueError(f"Unsupported explicit hint {policy} for mask {supported_mask}")
        vram_hint = requested
    else:
        raise ValueError(f"Invalid policy {policy}")

    passed_hint_field = vram_hint != UCS_MEMORY_TYPE_UNKNOWN
    effective_mem_type = vram_hint if passed_hint_field else ucx_detected_without_hint

    if effective_mem_type == UCS_MEMORY_TYPE_HOST:
        return UcxVramRegResult(
            scenario=scenario,
            policy=policy,
            supported_mask=supported_mask,
            resolved_hint=MEMTYPE_NAMES[vram_hint],
            passed_hint_field_to_ucp=passed_hint_field,
            detected_mem_type=MEMTYPE_NAMES[effective_mem_type],
            registration_succeeded=False,
            unmapped_on_host_misdetection=True,
        )
    return UcxVramRegResult(
        scenario=scenario,
        policy=policy,
        supported_mask=supported_mask,
        resolved_hint=MEMTYPE_NAMES[vram_hint],
        passed_hint_field_to_ucp=passed_hint_field,
        detected_mem_type=MEMTYPE_NAMES[effective_mem_type],
        registration_succeeded=True,
        unmapped_on_host_misdetection=False,
    )


@dataclass(frozen=True)
class LibfabricCudaCtxResult:
    allocation_kind: str
    pointer_attr_ctx_is_null: bool
    legacy_update_ctx_ret: int
    legacy_bound_primary_ctx: bool
    legacy_fi_mr_regattr_ok: bool
    patched_update_ctx_ret: int
    patched_bound_primary_ctx: bool
    patched_fi_mr_regattr_ok: bool


def simulate_libfabric_cuda_ctx_binding(
    allocation_kind: str,
    pointer_attr_ctx_is_null: bool,
) -> LibfabricCudaCtxResult:
    """Model nixlLibfabricCudaCtx::cudaUpdateCtxPtr and nixlLibfabricEngine::regMem."""
    # Legacy (pre-1f4afe6b60): returns 0 even when ctx == nullptr!
    legacy_ret = 0
    legacy_bound_primary = False
    legacy_mr_ok = not pointer_attr_ctx_is_null

    # Patched (commit 1f4afe6b60): if (ctx == nullptr) return -1; -> triggers cudaSetDevice(mem.devId)
    patched_ret = -1 if pointer_attr_ctx_is_null else 0
    patched_bound_primary = pointer_attr_ctx_is_null
    patched_mr_ok = True

    return LibfabricCudaCtxResult(
        allocation_kind=allocation_kind,
        pointer_attr_ctx_is_null=pointer_attr_ctx_is_null,
        legacy_update_ctx_ret=legacy_ret,
        legacy_bound_primary_ctx=legacy_bound_primary,
        legacy_fi_mr_regattr_ok=legacy_mr_ok,
        patched_update_ctx_ret=patched_ret,
        patched_bound_primary_ctx=patched_bound_primary,
        patched_fi_mr_regattr_ok=patched_mr_ok,
    )


def simulate_notif_consumption(payload: bytes, num_polls: int = 3) -> dict:
    """Compare legacy `if message:` vs patched `if message is not None:` in check_remote_xfer_done."""
    legacy_queue = [payload]
    legacy_hits = 0
    for _ in range(num_polls):
        message = None
        found = False
        for msg in legacy_queue:
            if payload in msg:
                message = msg
                found = True
                break
        if message:  # BUG when message == b""
            legacy_queue.remove(message)
        if found:
            legacy_hits += 1

    patched_queue = [payload]
    patched_hits = 0
    for _ in range(num_polls):
        message = None
        found = False
        for msg in patched_queue:
            if payload in msg:
                message = msg
                found = True
                break
        if message is not None:  # Fixed in commit b55e79152f
            patched_queue.remove(message)
        if found:
            patched_hits += 1

    return {
        "payload_len": len(payload),
        "num_polls": num_polls,
        "legacy_hits": legacy_hits,
        "legacy_remaining_notifs": len(legacy_queue),
        "patched_hits": patched_hits,
        "patched_remaining_notifs": len(patched_queue),
    }


def build_summary() -> dict:
    cuda_only_mask = ucs_bit(UCS_MEMORY_TYPE_HOST) | ucs_bit(UCS_MEMORY_TYPE_CUDA) | ucs_bit(UCS_MEMORY_TYPE_CUDA_MANAGED)
    ze_only_mask = ucs_bit(UCS_MEMORY_TYPE_HOST) | ucs_bit(UCS_MEMORY_TYPE_ZE_DEVICE)
    mixed_cuda_ze_mask = cuda_only_mask | ucs_bit(UCS_MEMORY_TYPE_ZE_DEVICE)

    ucx_scenarios = [
        simulate_ucx_vram_registration(
            scenario="cuda_auto_leaves_hint_unknown",
            policy="auto",
            supported_mask=cuda_only_mask,
            ucx_detected_without_hint=UCS_MEMORY_TYPE_CUDA,
        ),
        simulate_ucx_vram_registration(
            scenario="ze_only_auto_hints_ze_device",
            policy="auto",
            supported_mask=ze_only_mask,
            ucx_detected_without_hint=UCS_MEMORY_TYPE_HOST,
        ),
        simulate_ucx_vram_registration(
            scenario="mixed_cuda_ze_auto_conservative_unknown",
            policy="auto",
            supported_mask=mixed_cuda_ze_mask,
            ucx_detected_without_hint=UCS_MEMORY_TYPE_CUDA,
        ),
        simulate_ucx_vram_registration(
            scenario="cuda_managed_explicit_hint_recovers_host_misdetect",
            policy="cuda-managed",
            supported_mask=cuda_only_mask,
            ucx_detected_without_hint=UCS_MEMORY_TYPE_HOST,
        ),
        simulate_ucx_vram_registration(
            scenario="unhinted_host_misdetect_aborts_and_unmaps",
            policy="none",
            supported_mask=cuda_only_mask,
            ucx_detected_without_hint=UCS_MEMORY_TYPE_HOST,
        ),
    ]

    libfabric_cases = [
        simulate_libfabric_cuda_ctx_binding("cudaMalloc_primary_ctx", pointer_attr_ctx_is_null=False),
        simulate_libfabric_cuda_ctx_binding("cuMemCreate_vmm_expandable", pointer_attr_ctx_is_null=True),
    ]

    notif_cases = {
        "non_empty_tag": simulate_notif_consumption(b"kv_block_42", num_polls=3),
        "empty_barrier_notif": simulate_notif_consumption(b"", num_polls=3),
    }

    return {
        "ucx_scenarios": {s.scenario: asdict(s) for s in ucx_scenarios},
        "libfabric_cases": {c.allocation_kind: asdict(c) for c in libfabric_cases},
        "notif_cases": notif_cases,
    }


def verify_summary(summary: dict) -> None:
    ucx = summary["ucx_scenarios"]
    assert ucx["cuda_auto_leaves_hint_unknown"]["resolved_hint"] == "unknown"
    assert ucx["cuda_auto_leaves_hint_unknown"]["passed_hint_field_to_ucp"] is False
    assert ucx["cuda_auto_leaves_hint_unknown"]["registration_succeeded"] is True

    assert ucx["ze_only_auto_hints_ze_device"]["resolved_hint"] == "ze-device"
    assert ucx["ze_only_auto_hints_ze_device"]["passed_hint_field_to_ucp"] is True
    assert ucx["ze_only_auto_hints_ze_device"]["registration_succeeded"] is True

    assert ucx["mixed_cuda_ze_auto_conservative_unknown"]["resolved_hint"] == "unknown"
    assert ucx["cuda_managed_explicit_hint_recovers_host_misdetect"]["resolved_hint"] == "cuda-managed"
    assert ucx["cuda_managed_explicit_hint_recovers_host_misdetect"]["registration_succeeded"] is True

    assert ucx["unhinted_host_misdetect_aborts_and_unmaps"]["registration_succeeded"] is False
    assert ucx["unhinted_host_misdetect_aborts_and_unmaps"]["unmapped_on_host_misdetection"] is True

    lf = summary["libfabric_cases"]["cuMemCreate_vmm_expandable"]
    assert lf["legacy_fi_mr_regattr_ok"] is False
    assert lf["patched_update_ctx_ret"] == -1
    assert lf["patched_bound_primary_ctx"] is True
    assert lf["patched_fi_mr_regattr_ok"] is True

    empty_notif = summary["notif_cases"]["empty_barrier_notif"]
    assert empty_notif["legacy_hits"] == 3 and empty_notif["legacy_remaining_notifs"] == 1
    assert empty_notif["patched_hits"] == 1 and empty_notif["patched_remaining_notifs"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        verify_summary(summary)
    if args.json or not args.check:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("nixl_ucx_vram_vmm_notif_sim: CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
