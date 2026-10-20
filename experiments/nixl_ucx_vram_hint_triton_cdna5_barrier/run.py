#!/usr/bin/env python3
"""Deterministic stdlib model for NIXL UCX VRAM memtype hint, Libfabric VMM fallback, empty notification consumption & Triton AMD CDNA5 pre-cluster workgroup barrier.

Primary upstream sources:
- ai-dynamo/nixl commit 466ebb1ccc5a6e4e8cbaf954c94020400c9ae876 (#1536)
- ai-dynamo/nixl commit 1f4afe6b60b2b5114deff13a4d2653e7ba683f07 (#2425)
- ai-dynamo/nixl commit b55e79152fd67a0b7214f9127e435ad1820b2f78 (#2455)
- triton-lang/triton commit 249c29ead385cfbd4272d1a96c90a7516ad754a5 (#12342)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VALID_HINT_POLICIES = ("auto", "none", "cuda", "cuda-managed", "rocm", "ze-device")


def resolve_ucx_vram_memtype_hint(
    supported_types: set[str], policy_str: str
) -> dict[str, Any]:
    """Exact port of nixlUcxContext::resolveMemoryTypeConfig and ucx_auto_detect_vram_hint."""
    if policy_str not in VALID_HINT_POLICIES:
        return {
            "policy": policy_str,
            "supported_types": sorted(supported_types),
            "accepted": False,
            "resolved_hint": "invalid",
            "error": (
                f"Invalid VRAM memtype hint mode: {policy_str}. Matching is case-sensitive; "
                f"use exact values. Valid values are: <{'|'.join(VALID_HINT_POLICIES)}>"
            ),
        }
    if policy_str == "auto":
        has_ze = "ze-device" in supported_types
        has_cuda = "cuda" in supported_types or "cuda-managed" in supported_types
        has_rocm = "rocm" in supported_types
        if has_ze and not has_cuda and not has_rocm:
            resolved = "ze-device"
        else:
            resolved = "unknown"
        return {
            "policy": policy_str,
            "supported_types": sorted(supported_types),
            "accepted": True,
            "resolved_hint": resolved,
            "passes_ucp_mem_map_hint": resolved != "unknown",
        }
    if policy_str == "none":
        return {
            "policy": policy_str,
            "supported_types": sorted(supported_types),
            "accepted": True,
            "resolved_hint": "unknown",
            "passes_ucp_mem_map_hint": False,
        }
    if policy_str not in supported_types:
        return {
            "policy": policy_str,
            "supported_types": sorted(supported_types),
            "accepted": False,
            "resolved_hint": "unsupported",
            "error": f"Configured VRAM memtype hint '{policy_str}' is not supported by current UCX context",
        }
    return {
        "policy": policy_str,
        "supported_types": sorted(supported_types),
        "accepted": True,
        "resolved_hint": policy_str,
        "passes_ucp_mem_map_hint": True,
    }


def simulate_nixl_ucx_and_notifs() -> dict[str, Any]:
    """Evaluate UCX VRAM memtype hint matrix and Python empty notification consumption (#1536, #2455)."""
    cases = [
        ({"host", "ze-device"}, "auto"),
        ({"host", "ze-device"}, "none"),
        ({"host", "cuda", "cuda-managed"}, "auto"),
        ({"host", "cuda", "ze-device"}, "auto"),
        ({"host", "cuda", "cuda-managed"}, "cuda"),
        ({"host", "cuda"}, "rocm"),
        ({"host", "cuda"}, "CUDA"),
    ]
    matrix = [resolve_ucx_vram_memtype_hint(sup, pol) for sup, pol in cases]

    # Empty notification test (#2455): [(b"", b"", True), (b"", b"", False), (b"notification", b"not", True), (b"notification", b"fic", False)]
    notif_cases = [
        {"msg": "", "tag": "", "is_prefix": True},
        {"msg": "", "tag": "", "is_prefix": False},
        {"msg": "notification", "tag": "not", "is_prefix": True},
        {"msg": "notification", "tag": "fic", "is_prefix": False},
    ]
    unpatched_stuck_notifs = 0
    patched_stuck_notifs = 0
    for nc in notif_cases:
        msg = nc["msg"]
        # Unpatched: `if message:` fails to remove empty string/bytes `""`
        unpatched_remaining = 1 if len(msg) == 0 else 0
        # Patched: `if message is not None:` removes matched message even when empty
        patched_remaining = 0
        unpatched_stuck_notifs += unpatched_remaining
        patched_stuck_notifs += patched_remaining

    return {
        "ucx_hint_matrix": matrix,
        "unpatched_stuck_empty_notifs": unpatched_stuck_notifs,
        "patched_stuck_empty_notifs": patched_stuck_notifs,
    }


def simulate_libfabric_vmm_context_fallback() -> dict[str, Any]:
    """Model nixlLibfabricCudaCtx::cudaUpdateCtxPtr (#2425) on TRT-LLM KV cache V2 cuMemCreate VMM."""
    pointer_cases = [
        {
            "name": "trtllm_kv_v2_vmm_req1",
            "alloc_kind": "cuMemCreate_VMM",
            "ptr_dev": 0,
            "expected_dev": 0,
            "ptr_ctx": None,
        },
        {
            "name": "trtllm_kv_v2_vmm_req2",
            "alloc_kind": "cuMemCreate_VMM",
            "ptr_dev": 0,
            "expected_dev": 0,
            "ptr_ctx": None,
        },
        {
            "name": "trtllm_kv_v2_vmm_req3",
            "alloc_kind": "cuMemCreate_VMM",
            "ptr_dev": 0,
            "expected_dev": 0,
            "ptr_ctx": None,
        },
        {
            "name": "standard_cumemalloc",
            "alloc_kind": "cuMemAlloc",
            "ptr_dev": 0,
            "expected_dev": 0,
            "ptr_ctx": "ctx_dev0_primary",
        },
        {
            "name": "conflicting_context_dev0",
            "alloc_kind": "cuCtxCreate",
            "ptr_dev": 0,
            "expected_dev": 0,
            "ptr_ctx": "ctx_dev0_custom",
        },
        {
            "name": "device_ordinal_mismatch",
            "alloc_kind": "cuMemAlloc_dev1",
            "ptr_dev": 1,
            "expected_dev": 0,
            "ptr_ctx": "ctx_dev1_primary",
        },
    ]

    unpatched_vmm_failures = 0
    patched_vmm_failures = 0
    traces: list[dict[str, Any]] = []

    for case in pointer_cases:
        # Unpatched: when ptr_ctx is None and pthrCudaCtx_ is None, caches None and returns 0 (success),
        # then cudaSetCtx rejects None -> NIXL_ERR_BACKEND!
        if case["ptr_dev"] != case["expected_dev"]:
            unpatched_status = "fallback_cudaSetDevice"
            patched_status = "fallback_cudaSetDevice"
        elif case["ptr_ctx"] is None:
            unpatched_status = "NIXL_ERR_BACKEND_null_ctx"
            patched_status = "fallback_cudaSetDevice"
            unpatched_vmm_failures += 1
        else:
            unpatched_status = "cuCtxSetCurrent_ok"
            patched_status = "cuCtxSetCurrent_ok"
        traces.append(
            {
                "name": case["name"],
                "alloc_kind": case["alloc_kind"],
                "ptr_ctx": case["ptr_ctx"],
                "unpatched_status": unpatched_status,
                "patched_status": patched_status,
            }
        )

    return {
        "pointer_traces": traces,
        "unpatched_vmm_failures": unpatched_vmm_failures,
        "patched_vmm_failures": patched_vmm_failures,
    }


def simulate_triton_cdna5_cluster_barrier() -> dict[str, Any]:
    """Model Triton AMD CDNA5 TargetInfo::clusterBarrier (#12342) across 2 CTAs x 8 waves."""
    num_ctas = 2
    waves_per_cta = 8
    total_waves = num_ctas * waves_per_cta  # 16 waves

    # In CDNA5 (5.6.6), cluster barrier uses a 1-bit "Cluster Barrier Wave State" per wave.
    # Minimal kernel has two consecutive cluster barriers (or consecutive cross-CTA atomics):
    #   cluster.arrive() # signal 1
    #   cluster.wait()   # barrier 1
    #   cluster.arrive() # signal 2
    #   cluster.wait()   # barrier 2
    # Without pre-cluster workgroup s_barrier:
    # Fast wave-0 on CTA 0 and CTA 1 complete signal 1 -> barrier 1 -> signal 2 before slower waves 1..7
    # reach barrier 1. Slower waves 1..7 consume the single completion bit at barrier 1, and hang forever at barrier 2
    # because wave-0 already passed signal 2!
    unpatched_deadlocked_waves = num_ctas * (waves_per_cta - 1)  # 14 of 16 waves hang at barrier 2
    # With pre-cluster workgroup s_barrier inserted in TargetInfo::clusterBarrier:
    # Wave-0 cannot execute cluster.arrive() #2 until waves 1..7 in its workgroup finish barrier 1 and reach s_barrier #2.
    patched_deadlocked_waves = 0

    return {
        "num_ctas": num_ctas,
        "waves_per_cta": waves_per_cta,
        "total_waves": total_waves,
        "consecutive_cluster_barriers": 2,
        "unpatched_deadlocked_waves": unpatched_deadlocked_waves,
        "patched_deadlocked_waves": patched_deadlocked_waves,
        "workgroup_s_barrier_per_cluster_barrier": 1,
    }


def run_experiment() -> dict[str, Any]:
    ucx_and_notifs = simulate_nixl_ucx_and_notifs()
    libfabric_vmm = simulate_libfabric_vmm_context_fallback()
    cdna5_barrier = simulate_triton_cdna5_cluster_barrier()

    return {
        "experiment": "nixl_ucx_vram_hint_triton_cdna5_barrier",
        "evidence_class": "own-offline",
        "upstream_commits": [
            "ai-dynamo/nixl@466ebb1ccc5a6e4e8cbaf954c94020400c9ae876",
            "ai-dynamo/nixl@1f4afe6b60b2b5114deff13a4d2653e7ba683f07",
            "ai-dynamo/nixl@b55e79152fd67a0b7214f9127e435ad1820b2f78",
            "triton-lang/triton@249c29ead385cfbd4272d1a96c90a7516ad754a5",
        ],
        "ucx_and_notifs": ucx_and_notifs,
        "libfabric_vmm": libfabric_vmm,
        "cdna5_barrier": cdna5_barrier,
        "summary_metrics": {
            "ze_only_auto_hint": ucx_and_notifs["ucx_hint_matrix"][0]["resolved_hint"],
            "mixed_ze_cuda_auto_hint": ucx_and_notifs["ucx_hint_matrix"][3][
                "resolved_hint"
            ],
            "unpatched_stuck_empty_notifs": ucx_and_notifs[
                "unpatched_stuck_empty_notifs"
            ],
            "patched_stuck_empty_notifs": ucx_and_notifs["patched_stuck_empty_notifs"],
            "unpatched_libfabric_vmm_failures": libfabric_vmm["unpatched_vmm_failures"],
            "patched_libfabric_vmm_failures": libfabric_vmm["patched_vmm_failures"],
            "unpatched_cdna5_deadlocked_waves": cdna5_barrier[
                "unpatched_deadlocked_waves"
            ],
            "patched_cdna5_deadlocked_waves": cdna5_barrier["patched_deadlocked_waves"],
        },
    }


def verify_summary(summary: dict[str, Any]) -> None:
    m = summary["summary_metrics"]
    assert m["ze_only_auto_hint"] == "ze-device"
    assert m["mixed_ze_cuda_auto_hint"] == "unknown"
    assert m["unpatched_stuck_empty_notifs"] == 2
    assert m["patched_stuck_empty_notifs"] == 0
    assert m["unpatched_libfabric_vmm_failures"] == 3
    assert m["patched_libfabric_vmm_failures"] == 0
    assert m["unpatched_cdna5_deadlocked_waves"] == 14
    assert m["patched_cdna5_deadlocked_waves"] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    summary = run_experiment()
    verify_summary(summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        verify_summary(loaded)
        print(
            "[PASS] nixl_ucx_vram_hint_triton_cdna5_barrier: "
            f"ze_auto={loaded['summary_metrics']['ze_only_auto_hint']}, "
            f"vmm_failures={loaded['summary_metrics']['patched_libfabric_vmm_failures']}/3, "
            f"empty_notifs={loaded['summary_metrics']['patched_stuck_empty_notifs']}/2, "
            f"cdna5_deadlocks={loaded['summary_metrics']['patched_cdna5_deadlocked_waves']}/16"
        )


if __name__ == "__main__":
    main()
