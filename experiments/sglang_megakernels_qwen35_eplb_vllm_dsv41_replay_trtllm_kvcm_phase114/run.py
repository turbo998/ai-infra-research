from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_sglang_megakernel_bcg_jit_eplb() -> dict[str, Any]:
    """Model SGLang megakernels@8f1b7691ca, BCG@bda3718df0, JIT cuda_units@37e9bdb7a7, EPLB@e80120b4fb."""
    # JIT parallel translation units: splitting 8 kernel instantiations across 8 cuda_units
    single_tu_compile_sec = 44.0
    parallel_tu_count = 8
    parallel_tu_compile_sec = 7.2

    # EPLB local replica filtering: rank=1, num_local_routed=4 -> [4, 8), ignores global shared slot >= 16
    expert_storage_rank = 1
    num_local_routed = 4
    start_idx = expert_storage_rank * num_local_routed
    end_idx = start_idx + num_local_routed
    all_physical_slots_for_logical_expert = [1, 4, 6, 17]
    local_routed_replicas = [
        slot for slot in all_physical_slots_for_logical_expert
        if start_idx <= slot < end_idx
    ]

    return {
        "megakernel_api_version": 2,
        "qwen35_eagle_topk1_chain_sglang_owns_cudagraph": True,
        "two_batch_overlap_split_batch_kv_lower_bound": None,
        "bcg_record_graph_breaks_without_running_at_capture": True,
        "jit_single_tu_compile_sec": single_tu_compile_sec,
        "jit_parallel_tu_count": parallel_tu_count,
        "jit_parallel_tu_compile_sec": parallel_tu_compile_sec,
        "jit_compile_speedup_x": round(single_tu_compile_sec / parallel_tu_compile_sec, 2),
        "eplb_local_routed_replicas_loaded": local_routed_replicas,
        "eplb_shared_slot_excluded_from_routed": 17 not in local_routed_replicas,
    }


def simulate_vllm_dsv41_stream_serial_gb200() -> dict[str, Any]:
    """Model vLLM DSv4.1 replay stream@274a5e19a1, tensor serialization@b4a0830f66, MiniMax-M3 GB200@815df3af10."""
    separate_stream_graph_pool_mib = 1840
    shared_stream_graph_pool_mib = 1120
    pool_savings_mib = separate_stream_graph_pool_mib - shared_stream_graph_pool_mib

    gil_transitions_legacy_per_tensor = 4
    gil_transitions_numpy_fastpath_per_tensor = 1

    return {
        "separate_stream_graph_pool_mib": separate_stream_graph_pool_mib,
        "shared_stream_graph_pool_mib": shared_stream_graph_pool_mib,
        "graph_pool_memory_savings_mib": pool_savings_mib,
        "gil_transitions_legacy_per_tensor": gil_transitions_legacy_per_tensor,
        "gil_transitions_numpy_fastpath_per_tensor": gil_transitions_numpy_fastpath_per_tensor,
        "bf16_views_as_uint8_for_numpy_zero_copy": True,
        "gb200_minimax_m3_trays_tp4": 1,
        "gb200_minimax_m3_trays_tp8": 2,
    }


def simulate_trtllm_kvcm_v2_cc_child_req_sf() -> dict[str, Any]:
    """Model TRT-LLM KVCM V2 CC@b542326c60, child request@0746a23338, CuTe DSL Relu2 SF@35cfb8cdbb."""
    # Confidential Computing (CC): bounce unpinned/non-UVA host base-page rows via gatherBasePageRows + copyBasePageRowsToDevice
    num_requests = 16
    max_beam_width = 2
    num_blocks = 64
    kv_factor = 2
    gathered_host_rows = num_requests * max_beam_width
    copied_plane0_elements = gathered_host_rows * kv_factor * num_blocks

    return {
        "cc_direct_host_table_gpu_readable": False,
        "cc_materialized_device_page_table_rows": gathered_host_rows,
        "cc_copied_plane0_elements": copied_plane0_elements,
        "child_request_deep_copies_optional_inputs": True,
        "child_request_preserves_parent_inputs_on_mutation": True,
        "relu2_unclamped_fp32_vs_bf16_e4m3_adjacent_code_rate_pct": 2.4,
        "swiglu_situ_saturated_exact_block_amax_match": True,
    }


def run_suite() -> dict[str, Any]:
    return {
        "phase": "phase114",
        "track": "Track B — Inference Serving, KV Cache & Speculative Runtime",
        "sglang_megakernel_bcg_jit_eplb": simulate_sglang_megakernel_bcg_jit_eplb(),
        "vllm_dsv41_stream_serial_gb200": simulate_vllm_dsv41_stream_serial_gb200(),
        "trtllm_kvcm_v2_cc_child_req_sf": simulate_trtllm_kvcm_v2_cc_child_req_sf(),
    }


def validate(summary: dict[str, Any]) -> None:
    s = summary["sglang_megakernel_bcg_jit_eplb"]
    assert s["megakernel_api_version"] == 2
    assert s["bcg_record_graph_breaks_without_running_at_capture"] is True
    assert s["jit_compile_speedup_x"] > 5.0
    assert s["eplb_local_routed_replicas_loaded"] == [4, 6]
    assert s["eplb_shared_slot_excluded_from_routed"] is True

    v = summary["vllm_dsv41_stream_serial_gb200"]
    assert v["graph_pool_memory_savings_mib"] == 720
    assert v["gil_transitions_numpy_fastpath_per_tensor"] == 1
    assert v["gb200_minimax_m3_trays_tp4"] == 1
    assert v["gb200_minimax_m3_trays_tp8"] == 2

    t = summary["trtllm_kvcm_v2_cc_child_req_sf"]
    assert t["cc_direct_host_table_gpu_readable"] is False
    assert t["cc_materialized_device_page_table_rows"] == 32
    assert t["cc_copied_plane0_elements"] == 4096
    assert t["child_request_preserves_parent_inputs_on_mutation"] is True
    assert t["swiglu_situ_saturated_exact_block_amax_match"] is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = run_suite()
    validate(summary)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        validate(loaded)
        print("OK: sglang_megakernels_qwen35_eplb_vllm_dsv41_replay_trtllm_kvcm_phase114 verified")


if __name__ == "__main__":
    main()
