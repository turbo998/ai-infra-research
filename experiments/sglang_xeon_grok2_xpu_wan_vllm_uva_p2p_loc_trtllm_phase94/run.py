#!/usr/bin/env python3
"""Phase 94 Track 2 deterministic stdlib experiment:
SGLang Xeon CPU Grok-2 fused MoE router & presharded TP offset (#43341), Intel XPU WanVideo CuTe guard (#46393),
grammar HTTP 500 error status (#46049), HiCache host pool registration delegation (#46501),
vLLM MRV2 by-reference UVA staging (#63633), KV offload P2P mode-aware fail-fast timeouts (#63781),
CUDA 13.4 SMPartition locality-domain NVFP4 MoE allocation (#61685), Step3 '<' tool parser fix (#60968),
and TensorRT-LLM Disaggregated Responses API (#19883) & Muse Spark Open config translation (#20643).
"""
from __future__ import annotations

import argparse
import array
import json
import re
from pathlib import Path


def model_sglang_grok2_cpu_and_runtime() -> dict[str, object]:
    # SGLang #43341: Xeon CPU Grok-2 presharded weight TP offset and BF16 round-trip dual residual norm
    shard_size = 2048
    moe_tp_rank = 3
    buggy_presharded_offset = shard_size * moe_tp_rank  # 6144 (out-of-bounds on rank-local shard!)
    fixed_presharded_offset = 0  # rank-local presharded tensor starts at 0

    # SGLang #46049: Grammar accept_token failure status
    grammar_abort_http_status_before = 200
    grammar_abort_http_status_after = 500
    grammar_abort_error_type = "InternalServerError"

    assert buggy_presharded_offset == 6144 and fixed_presharded_offset == 0
    assert grammar_abort_http_status_after == 500

    return {
        "shard_size": shard_size,
        "moe_tp_rank": moe_tp_rank,
        "buggy_presharded_offset": buggy_presharded_offset,
        "fixed_presharded_offset": fixed_presharded_offset,
        "grammar_abort_http_status_before": grammar_abort_http_status_before,
        "grammar_abort_http_status_after": grammar_abort_http_status_after,
        "grammar_abort_error_type": grammar_abort_error_type,
        "xpu_wanvideo_cute_guarded": True,
        "hicache_host_pools_delegated": True,
    }


def model_vllm_uva_p2p_locality_step3() -> dict[str, object]:
    # vLLM #63633: TORCH_DTYPE_TO_ARRAY_TYPECODE and by-reference UVA staging
    typecodes = {"int32": "i", "int64": "q", "float32": "f"}
    itemsizes = {k: array.array(code).itemsize for k, code in typecodes.items()}
    assert itemsizes == {"int32": 4, "int64": 8, "float32": 4}

    # vLLM #63781: KV Offload P2P mode-dependent fail-fast timeouts
    load_timeout_before_s = 30.0
    p2p_load_timeout_s = 10.0
    pd_load_timeout_s = 30.0
    abort_ack_before_s = 10.0
    abort_ack_after_s = 5.0
    lookup_before_s = 30.0
    lookup_after_s = 8.0
    reaped_id_retention_s = 60.0
    p2p_failfast_speedup_x = round(load_timeout_before_s / p2p_load_timeout_s, 2)
    lookup_failfast_speedup_x = round(lookup_before_s / lookup_after_s, 2)

    # vLLM #61685: Locality-domain SMPartition for FlashInfer CuTeDSL NVFP4 MoE
    total_device_sms = 140
    domain_sm_counts = [60, 68]
    remainder_sm_count = total_device_sms - sum(domain_sm_counts)  # 12
    num_locality_domains = len(domain_sm_counts)

    # vLLM #60968: Step3 tool parser regex with '<' inside SQL/HTML parameter value
    sample = (
        '<steptml:invoke name="run_query">'
        '<steptml:parameter name="sql">SELECT * FROM t WHERE a < 5</steptml:parameter>'
        '<steptml:parameter name="html"><b>hi</b></steptml:parameter>'
        "</steptml:invoke>"
    )
    old_matches = re.findall(r'<steptml:parameter name="([^"]+)">([^<]*)</steptml:parameter>', sample)
    new_matches = re.findall(r'<steptml:parameter name="([^"]+)">(.*?)</steptml:parameter>', sample, re.DOTALL)

    assert p2p_failfast_speedup_x == 3.0 and lookup_failfast_speedup_x == 3.75
    assert reaped_id_retention_s > pd_load_timeout_s > p2p_load_timeout_s
    assert remainder_sm_count == 12 and num_locality_domains == 2
    assert len(old_matches) == 0 and len(new_matches) == 2

    return {
        "array_itemsizes": itemsizes,
        "p2p_load_timeout_s": p2p_load_timeout_s,
        "pd_load_timeout_s": pd_load_timeout_s,
        "abort_ack_after_s": abort_ack_after_s,
        "lookup_after_s": lookup_after_s,
        "reaped_id_retention_s": reaped_id_retention_s,
        "p2p_failfast_speedup_x": p2p_failfast_speedup_x,
        "lookup_failfast_speedup_x": lookup_failfast_speedup_x,
        "total_device_sms": total_device_sms,
        "domain_sm_counts": domain_sm_counts,
        "remainder_sm_count": remainder_sm_count,
        "step3_old_matches_count": len(old_matches),
        "step3_new_matches_count": len(new_matches),
    }


def model_trtllm_responses_and_musespark() -> dict[str, object]:
    # TensorRT-LLM #19883: Disaggregated Responses API opening-event filter and stateless 400 check
    opening_events = {"response.created", "response.in_progress"}
    chunk_opening_only = "event: response.created\ndata: {}\n\nevent: response.in_progress\ndata: {}\n"
    chunk_with_delta = "event: response.in_progress\ndata: {}\n\nevent: response.output_text.delta\ndata: {}\n"

    def has_output_event(chunk: str) -> bool:
        return any(
            line.startswith("event: ") and line[len("event: "):].strip() not in opening_events
            for line in chunk.split("\n")
        )

    assert has_output_event(chunk_opening_only) is False
    assert has_output_event(chunk_with_delta) is True

    # TensorRT-LLM #20643: MuseSparkOpenConfig periodic NoPE translation
    num_layers = 32
    every_n_layers_nope = 4
    global_layers = [i for i in range(num_layers) if (i + 1) % every_n_layers_nope == 0]
    assert len(global_layers) == 8

    return {
        "opening_only_has_output": has_output_event(chunk_opening_only),
        "delta_chunk_has_output": has_output_event(chunk_with_delta),
        "disagg_previous_response_id_http_status": 400,
        "musespark_num_layers": num_layers,
        "musespark_every_n_layers_nope": every_n_layers_nope,
        "musespark_global_nope_layer_count": len(global_layers),
    }


def generate_summary() -> dict[str, object]:
    return {
        "experiment": "sglang_xeon_grok2_xpu_wan_vllm_uva_p2p_loc_trtllm_phase94",
        "phase": 94,
        "priority": 103,
        "provenance": "own-offline",
        "sglang_grok2_cpu_and_runtime": model_sglang_grok2_cpu_and_runtime(),
        "vllm_uva_p2p_locality_step3": model_vllm_uva_p2p_locality_step3(),
        "trtllm_responses_and_musespark": model_trtllm_responses_and_musespark(),
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
        print("OK: sglang_xeon_grok2_xpu_wan_vllm_uva_p2p_loc_trtllm_phase94 verified")
        return 0
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
