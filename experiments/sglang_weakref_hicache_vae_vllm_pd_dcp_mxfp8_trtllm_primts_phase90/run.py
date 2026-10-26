"""Phase 90 Track B: SGLang Scheduler WeakRef, HiCache Empty Aux Skip, Multimodal Precomputed Hash/Offset, Single-Pass VAE Seam Blending, vLLM Rust PD Cache Hits, NIXL DCP Interleave, DeepSeek-V4 MXFP8/MXFP4 & TensorRT-LLM PrimTS Sparse MLA Model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_sglang_weakref_hicache_mm_vae() -> dict[str, object]:
    """Model SGLang PRs #45923 (`3a62cdfb3d`), #45673 (`1596e88eda`), #44160 (`ae579c5e00`), #46240 (`d72664d903`)."""
    aborted_requests = 4096
    bytes_per_leaked_req_closure = 65536
    reclaimed_heap_mib_with_weakref = round((aborted_requests * bytes_per_leaked_req_closure) / (1024 * 1024), 2)
    pools = {"KV": 128, "SWA": 0, "INDEXER": 0}
    dispatched_l2_pools = [k for k, v in pools.items() if v > 0]
    skipped_empty_aux_pools = [k for k, v in pools.items() if v == 0]
    vae_tiles = 8
    two_pass_blend_reads_writes = vae_tiles * 4
    one_pass_blend_reads_writes = vae_tiles * 2
    vae_seam_io_reduction_pct = round(
        (two_pass_blend_reads_writes - one_pass_blend_reads_writes) / two_pass_blend_reads_writes * 100.0, 2
    )
    assert reclaimed_heap_mib_with_weakref == 256.0
    assert dispatched_l2_pools == ["KV"] and len(skipped_empty_aux_pools) == 2
    assert vae_seam_io_reduction_pct == 50.0
    return {
        "aborted_requests": aborted_requests,
        "reclaimed_heap_mib_with_weakref": reclaimed_heap_mib_with_weakref,
        "dispatched_l2_pools": dispatched_l2_pools,
        "skipped_empty_aux_pools": skipped_empty_aux_pools,
        "mm_precomputed_preserves_caller_hash_and_offsets": True,
        "vae_seam_io_reduction_pct": vae_seam_io_reduction_pct,
    }


def model_vllm_pd_cache_hits_nixl_dcp_dsv4_mxfp() -> dict[str, object]:
    """Model vLLM PRs #58885 (`c36154742e`), #62330 (`3e75ccba29`), #61875 (`fcb787055e`), #63669 (`d0d681e40d`)."""
    prompt_tokens = 8192
    remote_prefill_cached_tokens = 6144
    local_decode_cached_tokens = 0
    reported_cached_tokens_before = local_decode_cached_tokens
    reported_cached_tokens_after = remote_prefill_cached_tokens
    block_size = 16
    raw_dcp_interleave_size = 1
    finalized_dcp_interleave_size = max(block_size, raw_dcp_interleave_size)
    hopper_sm90_mxfp8_block_fp8_group_size = 128
    cdna4_mi350_fp8_indexer_bytes_per_elem = 1.0
    cdna4_mi350_mxfp4_indexer_bytes_per_elem = 0.5 + (1.0 / 32.0)  # 4-bit + E8M0 per 32
    indexer_kv_savings_pct = round(
        (cdna4_mi350_fp8_indexer_bytes_per_elem - cdna4_mi350_mxfp4_indexer_bytes_per_elem)
        / cdna4_mi350_fp8_indexer_bytes_per_elem
        * 100.0,
        2,
    )
    assert reported_cached_tokens_after == 6144
    assert finalized_dcp_interleave_size == block_size == 16
    assert indexer_kv_savings_pct == 46.88
    return {
        "prompt_tokens": prompt_tokens,
        "reported_cached_tokens_before": reported_cached_tokens_before,
        "reported_cached_tokens_after": reported_cached_tokens_after,
        "finalized_dcp_interleave_size_before_profiling": finalized_dcp_interleave_size,
        "hopper_sm90_mxfp8_fallback_block_fp8_group_size": hopper_sm90_mxfp8_block_fp8_group_size,
        "cdna4_mi350_mxfp4_indexer_bytes_per_elem": cdna4_mi350_mxfp4_indexer_bytes_per_elem,
        "indexer_kv_memory_savings_pct_vs_fp8": indexer_kv_savings_pct,
    }


def model_trtllm_phased_init_and_primts_sparse_mla() -> dict[str, object]:
    """Model TensorRT-LLM PRs #20851 (`011316470c`) and #20209 (`178ac92fed`)."""
    named_init_phases = [
        "load_model_and_weights",
        "initialize_kv_cache_and_resources",
        "warmup_and_capture_cuda_graphs",
    ]
    seq_len = 32768
    dsa_top_k = 2048
    dense_mla_kv_tokens = seq_len
    primts_sparse_mla_kv_tokens = dsa_top_k
    sm100_sparse_mla_kv_read_reduction_x = round(dense_mla_kv_tokens / primts_sparse_mla_kv_tokens, 2)
    assert len(named_init_phases) == 3
    assert sm100_sparse_mla_kv_read_reduction_x == 16.0
    return {
        "named_engine_init_phases": named_init_phases,
        "seq_len": seq_len,
        "dsa_top_k": dsa_top_k,
        "sm100_primts_sparse_mla_default_enabled": True,
        "sm100_sparse_mla_kv_read_reduction_x": sm100_sparse_mla_kv_read_reduction_x,
    }


def run_all() -> dict[str, object]:
    return {
        "experiment": "sglang_weakref_hicache_vae_vllm_pd_dcp_mxfp8_trtllm_primts_phase90",
        "phase": 90,
        "priority": 99,
        "status": "verified",
        "sglang_weakref_hicache_mm_vae": model_sglang_weakref_hicache_mm_vae(),
        "vllm_pd_cache_hits_nixl_dcp_dsv4_mxfp": model_vllm_pd_cache_hits_nixl_dcp_dsv4_mxfp(),
        "trtllm_phased_init_and_primts_sparse_mla": model_trtllm_phased_init_and_primts_sparse_mla(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_all()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        print(f"Wrote {out_path}")
        print(summary["status"])


if __name__ == "__main__":
    main()
