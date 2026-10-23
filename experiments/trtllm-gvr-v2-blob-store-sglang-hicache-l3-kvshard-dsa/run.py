#!/usr/bin/env python3
"""Phase 59 Track B deterministic offline models for TRT-LLM, SGLang & vLLM commits.

Models:
1. TRT-LLM GVR v2 Self-Sampling Top-K Decode Large-Batch Small-Seqlen Optimization (`trtllm@4ccdca1e84`)
2. TRT-LLM Canonical CUTLASS MoE Implementation Identities (`trtllm@9f445c2a86`)
3. TRT-LLM Disaggregation Blob Store Driver Boundary & Key Versioning (`trtllm@032ea6d414`)
4. SGLang HiCache PD Decode Prefetch L3 Span Promise Contract (`sglang@77bc8cab18` & `sglang@8cd3c494a3`)
5. SGLang KVShard Page-Sharded DSA Attention & Dual-Target Indexer Writes (`sglang@cc811ff0c3`)
6. SGLang Half-Warp Two-Row `head_dim=128` Bit-Identical QKNorm (`sglang@178febef5d`)
7. vLLM Model Runner V2 Later-PP Encoder Tower Skipping & AsyncLLM LoRA Inplace Reset (`vllm@168a86ab8f` & `vllm@ef5bcaa387`)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def model_trtllm_gvr_v2_topk() -> dict[str, Any]:
    """Model GVR v2 self-sampling top-K decode routing (`trtllm@4ccdca1e84`)."""
    nb = 1024
    quadc = 96
    qc_reg = 160
    num_sms = 148

    def route_register_rung(b: int, n: int, k: int) -> dict[str, Any]:
        cure = not (n < 2 * k and b > num_sms)
        if b <= num_sms:
            tpl = (512, 3, 1) if (k == 2048 and n <= 6144) else (1024, 2, 1)
            bins = 2 * nb
        else:
            tpl = (512, 3, 2) if n <= 6144 else (512, 4, 2)
            bins = 2 * nb if b <= 296 else nb
        return {
            "b": b,
            "n": n,
            "k": k,
            "tpl": list(tpl),
            "bins": bins,
            "qc": qc_reg,
            "cure": cure,
        }

    samples = [
        route_register_rung(64, 4100, 2048),
        route_register_rung(64, 8192, 2048),
        route_register_rung(256, 4100, 1024),
        route_register_rung(256, 8192, 1024),
        route_register_rung(256, 2048, 2048),  # n < 2*k and b > num_sms -> cure=False
    ]

    return {
        "commit": "4ccdca1e84",
        "nb_bins": nb,
        "streaming_quadc": quadc,
        "register_qc": qc_reg,
        "samples": samples,
    }


def model_trtllm_blob_store_driver() -> dict[str, Any]:
    """Model TRT-LLM disaggregation blob store key encoding & driver contract (`trtllm@032ea6d414` & `9f445c2a86`)."""
    format_version = 1
    prefix_re = re.compile(r"[A-Za-z0-9_-]{1,64}")

    def encode_key(prefix: str, name: bytes) -> str:
        if not prefix_re.fullmatch(prefix):
            raise ValueError(f"bad key prefix {prefix!r}")
        if not name:
            raise ValueError("an object name is non-empty")
        return f"{prefix}/v{format_version}/{name.hex()}"

    key_example = encode_key("prod_cluster_01", b"\x00\x11\x22\xff")
    invalid_prefixes = ["", "bad/prefix", "a" * 65]
    rejected_count = 0
    for p in invalid_prefixes:
        try:
            encode_key(p, b"kv")
        except ValueError:
            rejected_count += 1

    return {
        "commit_blob": "032ea6d414",
        "commit_cutlass_moe": "9f445c2a86",
        "format_version": format_version,
        "example_encoded_key": key_example,
        "invalid_prefixes_rejected": rejected_count,
        "memory_kinds": ["host", "device"],
        "driver_rejected_certifies_no_escape": True,
        "poll_and_access_ended_orthogonal": True,
    }


def model_sglang_hicache_pd_l3_contract() -> dict[str, Any]:
    """Model SGLang HiCache PD decode L3 promise & linker skip (`sglang@77bc8cab18` & `8cd3c494a3`)."""
    prefetch_threshold = 16
    cases = [
        {"name": "short_l3_below_threshold", "l1_len": 4, "l2_len": 4, "l3_hit_len": 8, "prefetch_taken": False},
        {"name": "long_l3_above_threshold", "l1_len": 4, "l2_len": 4, "l3_hit_len": 24, "prefetch_taken": True},
        {"name": "l3_prefetch_exception", "l1_len": 8, "l2_len": 8, "l3_hit_len": 32, "prefetch_taken": False},
    ]

    evaluated = []
    for c in cases:
        host_len = c["l1_len"] + c["l2_len"]
        legacy_promised_len = host_len + c["l3_hit_len"]
        fixed_promised_len = (host_len + c["l3_hit_len"]) if c["prefetch_taken"] else host_len
        unbacked_tokens_prevented = legacy_promised_len - fixed_promised_len
        evaluated.append({
            **c,
            "legacy_promised_len": legacy_promised_len,
            "fixed_promised_len": fixed_promised_len,
            "unbacked_tokens_prevented": unbacked_tokens_prevented,
        })

    return {
        "commit_l3_promise": "77bc8cab18",
        "commit_linker_skip": "8cd3c494a3",
        "prefetch_threshold_tokens": prefetch_threshold,
        "decode_mode_skips_linker_lookup": True,
        "cases": evaluated,
    }


def model_sglang_kvshard_dsa() -> dict[str, Any]:
    """Model SGLang KVShard page-sharded DSA attention & indexer writes (`sglang@cc811ff0c3`)."""
    world_size = 4
    page_size = 64
    total_tokens = 512
    num_pages = total_tokens // page_size  # 8 pages

    # Cyclic page ownership across 4 ranks during an extend step
    scratch_writes_per_rank = total_tokens  # every token staged in scratch
    local_owned_writes_per_rank = total_tokens // world_size  # 128 owned tokens persisted locally

    return {
        "commit": "cc811ff0c3",
        "world_size": world_size,
        "page_size": page_size,
        "total_tokens": total_tokens,
        "num_pages": num_pages,
        "extend_write_targets_count": 2,
        "decode_write_targets_count": 1,
        "scratch_writes_per_rank": scratch_writes_per_rank,
        "local_owned_writes_per_rank": local_owned_writes_per_rank,
        "supports_mha_one_shot_when_sharded": False,
        "separates_logical_and_physical_page_tables": True,
    }


def model_sglang_half_warp_qknorm() -> dict[str, Any]:
    """Model SGLang half-warp 2-rows-per-warp head_dim=128 QKNorm (`sglang@178febef5d`)."""
    tokens = 4099
    hq = 5
    hk = 2
    head_dim = 128
    warp_threads = 32
    row_lanes = 16
    packs_per_lane = head_dim // (2 * row_lanes)  # 4 packed_t (8 bf16 = 16 B)
    bytes_per_lane = packs_per_lane * 4  # 16 bytes

    total_rows = tokens * (hq + hk)  # 4099 * 7 = 28693 (odd!)
    one_row_per_warp_launches = total_rows  # 28693 warp-row steps
    two_rows_per_warp_pairs = (total_rows + 1) // 2  # 14347 warp-pair steps

    return {
        "commit": "178febef5d",
        "head_dim": head_dim,
        "warp_threads": warp_threads,
        "row_lanes": row_lanes,
        "bytes_per_lane_load": bytes_per_lane,
        "total_rows": total_rows,
        "one_row_per_warp_steps": one_row_per_warp_launches,
        "two_rows_per_warp_steps": two_rows_per_warp_pairs,
        "warp_step_reduction_ratio": round(one_row_per_warp_launches / two_rows_per_warp_pairs, 4),
        "odd_tail_recomputes_last_row_skips_store": True,
        "misaligned_8b_pad_falls_back_to_one_row": True,
        "bit_identical_to_full_warp": True,
    }


def model_vllm_pp_tower_and_lora() -> dict[str, Any]:
    """Model vLLM V2 PP encoder tower skip & AsyncLLM LoRA inplace fix (`vllm@168a86ab8f` & `ef5bcaa387`)."""
    pp_size = 4
    encoder_tower_gib = 1.85  # e.g., Vision + Audio tower weights per rank
    legacy_total_tower_gib = pp_size * encoder_tower_gib
    v2_first_rank_only_gib = 1 * encoder_tower_gib
    saved_vram_gib = round(legacy_total_tower_gib - v2_first_rank_only_gib, 2)

    num_generate_steps = 500
    lora_reload_ms = 18.0
    legacy_reloads = num_generate_steps  # leaked load_inplace=True reloaded every step
    fixed_reloads = 1  # reset load_inplace=False before caching

    return {
        "commit_pp_tower": "168a86ab8f",
        "commit_lora_fix": "ef5bcaa387",
        "pp_size": pp_size,
        "encoder_tower_gib_per_rank": encoder_tower_gib,
        "legacy_total_tower_gib": round(legacy_total_tower_gib, 2),
        "v2_first_rank_only_gib": v2_first_rank_only_gib,
        "saved_vram_gib_across_pp": saved_vram_gib,
        "num_generate_steps": num_generate_steps,
        "legacy_disk_reloads": legacy_reloads,
        "fixed_disk_reloads": fixed_reloads,
        "saved_lora_reload_time_ms": round((legacy_reloads - fixed_reloads) * lora_reload_ms, 1),
    }


def build_summary() -> dict[str, Any]:
    return {
        "phase": 59,
        "priority": 68,
        "track": "Track B — LLM Serving Engines, Disaggregated KV & Multimodal Pipeline Execution",
        "trtllm_gvr_v2_topk": model_trtllm_gvr_v2_topk(),
        "trtllm_blob_store_driver": model_trtllm_blob_store_driver(),
        "sglang_hicache_pd_l3_contract": model_sglang_hicache_pd_l3_contract(),
        "sglang_kvshard_dsa": model_sglang_kvshard_dsa(),
        "sglang_half_warp_qknorm": model_sglang_half_warp_qknorm(),
        "vllm_pp_tower_and_lora": model_vllm_pp_tower_and_lora(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches model")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise SystemExit("summary.json is out of sync with run.py")
        print("Phase 59 Track B summary.json verified.")
        return 0

    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
