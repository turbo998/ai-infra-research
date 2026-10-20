from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def analyze_mooncake_page_gather_transfer(
    num_fields: int = 64,
    num_pages: int = 512,
    max_batch_size: int = 8192,
) -> dict[str, Any]:
    """Model Mooncake PR #4763 (387aecb2) buffer-protocol and page-gathered batch transfer."""
    total_descriptors = num_fields * num_pages
    # List-based pybind11 API boxes 3 Python lists of length F*P (buffers, peer_addrs, lengths)
    legacy_python_int_conversions = 3 * total_descriptors

    # Buffer-protocol _arrays API passes 3 1-D contiguous uint64/int64 buffers (0 per-element pybind11 boxing)
    arrays_api_buffer_bytes = 3 * total_descriptors * 8

    # Page-gathered _pages API passes src_pages[P], dst_pages[P], and fields[F, 5] (8-byte integers)
    pages_api_input_ints = 2 * num_pages + 5 * num_fields
    pages_api_input_bytes = pages_api_input_ints * 8
    input_volume_reduction_ratio = 1.0 - (pages_api_input_ints / legacy_python_int_conversions)

    # Verify field-major descriptor expansion and max_batch_size chunking on a 2x3 example
    sample_src_pages = [10, 25, 40]
    sample_dst_pages = [3, 7, 12]
    # [src_base, src_stride, dst_base, dst_stride, length]
    sample_fields = [
        [0x10000000, 4096, 0x20000000, 4096, 2048],
        [0x30000000, 8192, 0x40000000, 8192, 4096],
    ]
    expanded_src = []
    expanded_dst = []
    expanded_len = []
    for src_base, src_stride, dst_base, dst_stride, length in sample_fields:
        for s_page, d_page in zip(sample_src_pages, sample_dst_pages):
            expanded_src.append(src_base + s_page * src_stride)
            expanded_dst.append(dst_base + d_page * dst_stride)
            expanded_len.append(length)

    num_batches = math.ceil(total_descriptors / max_batch_size) if max_batch_size > 0 else 1

    return {
        "num_fields": num_fields,
        "num_pages": num_pages,
        "total_descriptors": total_descriptors,
        "legacy_python_int_conversions": legacy_python_int_conversions,
        "arrays_api_buffer_bytes": arrays_api_buffer_bytes,
        "pages_api_input_ints": pages_api_input_ints,
        "pages_api_input_bytes": pages_api_input_bytes,
        "pages_api_input_reduction_pct": round(input_volume_reduction_ratio * 100.0, 4),
        "max_batch_size": max_batch_size,
        "submitted_rpc_batches": num_batches,
        "sample_field_major_src_addrs": expanded_src,
        "sample_field_major_dst_addrs": expanded_dst,
        "sample_field_major_lengths": expanded_len,
    }


def analyze_lmcache_scratch_group_skip() -> dict[str, Any]:
    """Model LMCache PR #5818 (2a34fa82) LazyOffloadManager scratch group span-0 skip."""
    cases = [
        {"group_tokens_per_block": [16, 0, 32], "scheduled_tokens": 48, "expected_blocks": 5},
        {"group_tokens_per_block": [16, 0, 32], "scheduled_tokens": 17, "expected_blocks": 3},
        {"group_tokens_per_block": [0, 64], "scheduled_tokens": 65, "expected_blocks": 2},
    ]
    results = []
    for case in cases:
        spans = case["group_tokens_per_block"]
        tokens = case["scheduled_tokens"]
        legacy_raises_zero_division = 0 in spans
        est_blocks = sum(-(-tokens // s) for s in spans if s > 0)
        results.append(
            {
                "group_tokens_per_block": spans,
                "scheduled_tokens": tokens,
                "legacy_raises_zero_division": legacy_raises_zero_division,
                "est_next_step_blocks": est_blocks,
                "expected_blocks": case["expected_blocks"],
            }
        )
    return {"cases": results}


def analyze_dynamo_multi_connector_kvcr_tiers() -> dict[str, Any]:
    """Model NVIDIA Dynamo PR #16778 (662cb22d) MultiConnector secondary_tiers discovery."""
    multi_connector_wrappers = {"MultiConnector"}
    kv_config = {
        "kv_connector": "MultiConnector",
        "kv_connector_extra_config": {
            "connectors": [
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
                {
                    "kv_connector": "OffloadingConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "spec_name": "TieringOffloadingSpec",
                        "secondary_tiers": [
                            {
                                "type": "kvcr",
                                "router_capabilities": ["kv_hint_transfer"],
                                "control_advertise_host": "127.0.0.1",
                                "control_ports": [17771],
                            }
                        ],
                    },
                },
            ]
        },
    }

    extra_config = kv_config["kv_connector_extra_config"]
    # Legacy bug: only inspected top-level extra_config.get("secondary_tiers")
    legacy_tiers = extra_config.get("secondary_tiers") or []

    # Fixed PR #16778: unwraps child kv_connector_extra_config inside MULTI_CONNECTOR_WRAPPERS
    extra_configs = [extra_config]
    if kv_config.get("kv_connector") in multi_connector_wrappers:
        extra_configs = [
            child.get("kv_connector_extra_config") or {}
            for child in extra_config.get("connectors") or []
        ]
    fixed_tiers = []
    for cfg in extra_configs:
        tiers = cfg.get("secondary_tiers")
        if isinstance(tiers, list):
            fixed_tiers.extend(t for t in tiers if isinstance(t, dict))

    endpoints = {
        str(idx): f"tcp://{tier['control_advertise_host']}:{port}"
        for tier in fixed_tiers
        for idx, port in enumerate(tier.get("control_ports", []))
    }

    return {
        "legacy_discovered_tiers": len(legacy_tiers),
        "fixed_discovered_tiers": len(fixed_tiers),
        "kv_hint_transfer_enabled": any(
            "kv_hint_transfer" in t.get("router_capabilities", []) for t in fixed_tiers
        ),
        "published_control_endpoints": endpoints,
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "mooncake_page_gather_lmcache_scratch_dynamo_kvcr",
        "mooncake_page_gather": analyze_mooncake_page_gather_transfer(),
        "lmcache_scratch_skip": analyze_lmcache_scratch_group_skip(),
        "dynamo_multi_connector_kvcr": analyze_dynamo_multi_connector_kvcr_tiers(),
    }


def verify_summary(summary: dict[str, Any]) -> None:
    mc = summary["mooncake_page_gather"]
    assert mc["total_descriptors"] == 32768
    assert mc["legacy_python_int_conversions"] == 98304
    assert mc["pages_api_input_ints"] == 1344
    assert abs(mc["pages_api_input_reduction_pct"] - 98.6328) < 1e-3
    assert mc["submitted_rpc_batches"] == 4
    assert mc["sample_field_major_src_addrs"] == [
        0x10000000 + 10 * 4096,
        0x10000000 + 25 * 4096,
        0x10000000 + 40 * 4096,
        0x30000000 + 10 * 8192,
        0x30000000 + 25 * 8192,
        0x30000000 + 40 * 8192,
    ]

    lm = summary["lmcache_scratch_skip"]
    for case in lm["cases"]:
        assert case["legacy_raises_zero_division"] is True
        assert case["est_next_step_blocks"] == case["expected_blocks"]

    dyn = summary["dynamo_multi_connector_kvcr"]
    assert dyn["legacy_discovered_tiers"] == 0
    assert dyn["fixed_discovered_tiers"] == 1
    assert dyn["kv_hint_transfer_enabled"] is True
    assert dyn["published_control_endpoints"] == {"0": "tcp://127.0.0.1:17771"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    verify_summary(summary)

    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    res_dir = Path(__file__).resolve().parent / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        print("OK: mooncake_page_gather_lmcache_scratch_dynamo_kvcr verified")


if __name__ == "__main__":
    main()
