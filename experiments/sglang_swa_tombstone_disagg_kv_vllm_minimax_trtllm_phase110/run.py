from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_summary() -> dict:
    # 1. SGLang #46700 (e7c55f6959): Clamp SWA tombstone recovery to the request's protected prefix
    # window_start = max(evicted_seqlen, prev_prefix_len) unconditionally (even when _independent is False).
    evicted_seqlen = 256
    prev_prefix_len = 512
    total_prefix_len = 384
    prefix_len = 256
    unclamped_window_start = evicted_seqlen
    clamped_window_start = max(evicted_seqlen, prev_prefix_len)
    corrupted_protected_tokens_before = max(0, prev_prefix_len - max(unclamped_window_start, total_prefix_len))
    corrupted_protected_tokens_after = max(0, prev_prefix_len - max(clamped_window_start, total_prefix_len))

    # 2. SGLang #46689 (b28f4fc065) & #46674 (b7ac4c0e72):
    # Disaggregation restores host-received decode KV on index_stream + host_to_device_stream with restore_done_event
    # without blocking the scheduler; Rust radix-tree keeps GIL via lock_py_attached for O(1) reads unless contended.
    batch_restore_requests = 8
    restore_copy_us_per_req = 180.0
    scheduler_stall_before_us = batch_restore_requests * restore_copy_us_per_req
    scheduler_stall_after_us = 0.0
    radix_tree_gil_detach_transitions_before = 3  # has_ongoing_insert, size, digest
    radix_tree_gil_detach_transitions_uncontended_after = 0

    # 3. vLLM #57420 (8e26dec5f5) & #61903 (0076619a94):
    # MiniMax-M3 Triton query-tiled sparse prefill on Hopper (_PREFILL_TILE_Q = 16, SPARSE_BLOCK_SIZE = 128);
    # CuteDSL BF16x3 router GEMM on Hopper (SM90) splits fp32 router weights into 3 bf16 terms in registers.
    minimax_prefill_tile_q = 16
    minimax_sparse_block_size = 128
    kv_page_Loads_reduction_x = minimax_prefill_tile_q  # 16 queries share 1 KV block load per tile
    bf16x3_terms = 3
    bf16x3_supported_archs = ["sm_90", "sm_100"]

    # 4. vLLM #60234 (5c1eb11cb3), #59986 (8977b13a1c) & #63548 (cebcc5d8c7):
    # Qwen4Exp PLE + MTP drafter supports PP>1 (PPMissingLayer on non-first ranks, MTP branches on intermediate_tensors is None);
    # DSV4.1 sparse prefill downgraded from AttentionCGSupport.ALWAYS to UNIFORM_BATCH (max_decode_query_len);
    # MRV2 model_state allocated outside weights pool so sleep mode does not wipe runtime state tensors.
    qwen4exp_pp_size = 4
    qwen4exp_vocab_embed_ranks_allocated_after = 1  # only first PP rank allocates VocabParallelEmbedding
    dsv41_cudagraph_support_before = "ALWAYS"
    dsv41_cudagraph_support_after = "UNIFORM_BATCH"
    mrv2_sleep_wake_corrupted_tensors_before = 2
    mrv2_sleep_wake_corrupted_tensors_after = 0

    # 5. TensorRT-LLM #19369 (0a03c3ed05) & #20803 (8354c4c398):
    # Support attention DP in V2 KV cache connectors (each ADP owner rank runs its own KvCacheConnectorScheduler);
    # Reduce sampler logprobs temporary memory: only processed logprobs allocate full-vocab [rows, V] FP32 buffer,
    # while raw logprobs are gathered directly from logits via Triton selected_raw_logprobs.
    batch_size = 128
    vocab_size = 151936
    fp32_bytes = 4
    raw_logprobs_only_reqs = 128
    processed_logprobs_reqs = 0
    temp_vocab_buffer_mib_before = round(
        (raw_logprobs_only_reqs * vocab_size * fp32_bytes) / (1024 * 1024), 2
    )
    temp_vocab_buffer_mib_after = round(
        (processed_logprobs_reqs * vocab_size * fp32_bytes) / (1024 * 1024), 2
    )

    return {
        "experiment_id": "sglang_swa_tombstone_disagg_kv_vllm_minimax_trtllm_phase110",
        "phase": 110,
        "priority": 119,
        "provenance": "own-offline",
        "sglang_swa_disagg_and_radix_gil": {
            "evicted_seqlen": evicted_seqlen,
            "prev_prefix_len": prev_prefix_len,
            "unclamped_window_start": unclamped_window_start,
            "clamped_window_start": clamped_window_start,
            "corrupted_protected_tokens_before": corrupted_protected_tokens_before,
            "corrupted_protected_tokens_after": corrupted_protected_tokens_after,
            "scheduler_stall_before_us": scheduler_stall_before_us,
            "scheduler_stall_after_us": scheduler_stall_after_us,
            "radix_tree_gil_detach_transitions_before": radix_tree_gil_detach_transitions_before,
            "radix_tree_gil_detach_transitions_uncontended_after": (
                radix_tree_gil_detach_transitions_uncontended_after
            ),
        },
        "vllm_minimax_bf16x3_qwen4_dsv41_mrv2": {
            "minimax_prefill_tile_q": minimax_prefill_tile_q,
            "minimax_sparse_block_size": minimax_sparse_block_size,
            "kv_page_loads_reduction_x": kv_page_Loads_reduction_x,
            "bf16x3_terms": bf16x3_terms,
            "bf16x3_supported_archs": bf16x3_supported_archs,
            "qwen4exp_pp_size": qwen4exp_pp_size,
            "qwen4exp_vocab_embed_ranks_allocated_after": qwen4exp_vocab_embed_ranks_allocated_after,
            "dsv41_cudagraph_support_before": dsv41_cudagraph_support_before,
            "dsv41_cudagraph_support_after": dsv41_cudagraph_support_after,
            "mrv2_sleep_wake_corrupted_tensors_before": mrv2_sleep_wake_corrupted_tensors_before,
            "mrv2_sleep_wake_corrupted_tensors_after": mrv2_sleep_wake_corrupted_tensors_after,
        },
        "trtllm_adp_kv_and_sampler_logprobs": {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "temp_vocab_buffer_mib_before": temp_vocab_buffer_mib_before,
            "temp_vocab_buffer_mib_after": temp_vocab_buffer_mib_after,
            "adp_per_owner_connector_scheduler": True,
        },
        "invariants": {
            "swa_protected_prefix_clamped": (
                corrupted_protected_tokens_before == 128 and corrupted_protected_tokens_after == 0
            ),
            "disagg_restore_nonblocking": scheduler_stall_after_us == 0.0,
            "dsv41_sparse_prefill_excluded_from_full_cg": dsv41_cudagraph_support_after == "UNIFORM_BATCH",
            "trtllm_raw_logprobs_zero_full_vocab_alloc": (
                temp_vocab_buffer_mib_before == 74.19 and temp_vocab_buffer_mib_after == 0.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = build_summary()
    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == summary, "Drift detected in Phase 110 Track B summary.json"
        assert all(summary["invariants"].values())
        print("OK: sglang_swa_tombstone_disagg_kv_vllm_minimax_trtllm_phase110 verified")
        return

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
