from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any


def build_summary() -> dict[str, Any]:
    # 1. SGLang PD KV Adler-32 checksum & aux buffer guard (PR #39729, 5d11819f)
    prompt_pages = [b"kv_page_0_prefix", b"kv_page_1_prefix", b"kv_page_2_suffix"]
    full_prompt_adler32 = zlib.adler32(b"".join(prompt_pages))
    suffix_only_adler32 = zlib.adler32(prompt_pages[2])
    radix_prefix_hit_causes_false_mismatch = full_prompt_adler32 != suffix_only_adler32
    prefill_aux_buffers = 3
    decode_aux_buffers = 2
    transferred_aux_buffers = min(prefill_aux_buffers, decode_aux_buffers)

    # 2. SGLang LTX-2 vectorized Q/K RMSNorm + split-RoPE (PR #46111, a887ac8f)
    ltx2_num_heads = 32
    ltx2_head_dim = 128
    ltx2_half_dim = ltx2_head_dim // 2
    ltx2_num_pairs = ltx2_num_heads * ltx2_half_dim  # 2048 pairs per token
    ltx2_vec_width_pairs = 8  # 8 pairs = 16 bf16 = 32B (two 16B loads: x0_vec, x1_vec)
    ltx2_num_vecs_per_token = ltx2_num_pairs // ltx2_vec_width_pairs  # 256
    ltx2_tx_reduction_factor = ltx2_vec_width_pairs  # 8x fewer memory transactions

    # 3. SGLang QSA prefill compressed-key gather (PR #45752, dba22e9b)
    qsa_compress_ratio = 4
    qsa_seq_lengths = [131, 3, 4103, 64, 257]
    qsa_complete_blocks_per_seq = [n // qsa_compress_ratio for n in qsa_seq_lengths]
    qsa_total_complete_blocks = sum(qsa_complete_blocks_per_seq)  # 32 + 0 + 1025 + 16 + 64 = 1137
    qsa_legacy_launches = sum(1 for b in qsa_complete_blocks_per_seq if b > 0) + 1  # 4 index_select + 1 cat = 5
    qsa_new_triton_launches = 1

    # 4. SGLang EAGLE HiCache decode restore bigram boundary (PR #45076, 0ff67ee7)
    eagle_decode_prefix_bigrams = 64
    eagle_boundary_tokens = 1
    eagle_rematch_token_len = eagle_decode_prefix_bigrams + eagle_boundary_tokens  # 65

    # 5. vLLM ROCm 32-bit buffer addressing for TRITON_ATTN KV cache views (PR #60084, a55937f2)
    rocm_32bit_limit_bytes = 1 << 31  # 2,147,483,648 B (2 GiB)
    num_layers = 32
    layer_kv_bytes = 256 * 1024 * 1024  # 256 MiB per layer (128 MiB K + 128 MiB V)
    total_storage_bytes = num_layers * layer_kv_bytes  # 8 GiB > 2 GiB
    # Layer 0 view inside 8 GiB storage without ptr_range sees 8 GiB -> 64-bit addressing
    # With set_triton_ptr_range(kv_cache[layer], key_cache, value_cache), region is 256 MiB:
    layer_region_bytes = layer_kv_bytes
    key_view_ptr_range_bytes = layer_region_bytes  # 268,435,456 B < 2 GiB
    value_view_ptr_range_bytes = layer_region_bytes // 2  # 134,217,728 B < 2 GiB
    rocm_uses_32bit_buffer_ops_before = total_storage_bytes < rocm_32bit_limit_bytes
    rocm_uses_32bit_buffer_ops_after = (
        key_view_ptr_range_bytes < rocm_32bit_limit_bytes
        and value_view_ptr_range_bytes < rocm_32bit_limit_bytes
    )

    # 6. vLLM MRV2 DBO microbatch routed-expert offset & first-failure reporting (PR #63372 & #63476)
    dbo_ubatch0_tokens = 64
    dbo_ubatch1_tokens = 64
    dbo_ubatch_token_offsets = [0, dbo_ubatch0_tokens]
    dbo_captured_rows_before = dbo_ubatch1_tokens  # ubatch 1 overwrote [0:64]
    dbo_captured_rows_after = dbo_ubatch0_tokens + dbo_ubatch1_tokens  # [0:64] + [64:128] = 128
    # When ubatch 1 fails first and ubatch 0 fails later on recv hook:
    dbo_errorsinsertion_order = {1: "ValueError(ubatch1 root cause)", 0: "RuntimeError(recv_hook on exited ubatch1)"}
    dbo_reported_ubatch_before = min(dbo_errorsinsertion_order)  # 0 (wrong secondary error)
    dbo_reported_ubatch_after = next(iter(dbo_errorsinsertion_order))  # 1 (true root cause)

    # 7. vLLM SimpleCPUOffloadConnector chunked-local alignment guard (PR #63266, 65bc6983)
    attention_chunk_size = 64
    block_size = 16
    test_cases = [
        (0, 4, 64),
        (0, 5, 80),
        (16, 4, 0),
        (16, 5, 0),
        (64, 4, 0),
        (64, 5, 16),
    ]
    evaluated_cases = []
    for local_tokens, cached_blocks, expected_ext in test_cases:
        total_cached_tokens = cached_blocks * block_size
        if local_tokens % attention_chunk_size != 0:
            actual_ext = 0
        else:
            actual_ext = max(0, total_cached_tokens - local_tokens)
        assert actual_ext == expected_ext
        evaluated_cases.append(
            {
                "local_tokens": local_tokens,
                "cached_chunk_blocks": cached_blocks,
                "external_cpu_tokens": actual_ext,
            }
        )

    # 8. TensorRT-LLM MiniMax-M3 VL & LoRA ckpt_source (PR #20626 & #20774)
    minimax_bare_image_marker = "]&lt;]image[&gt;["
    minimax_bare_image_token = "]<]image[>["
    minimax_bare_video_token = "]<]video[>["
    minimax_chat_template_applied_flag = True
    lora_server_default_ckpt_source = "hf"
    lora_request_admitted_ckpt_source = "nemo"
    lora_reloaded_ckpt_source_before = lora_server_default_ckpt_source
    lora_reloaded_ckpt_source_after = lora_request_admitted_ckpt_source

    return {
        "experiment_id": "serving_pd_checksum_qsa_rocm_dbo_minimax_phase85",
        "phase": 85,
        "priority": 94,
        "sglang_pd_kv_checksum": {
            "full_prompt_adler32": full_prompt_adler32,
            "suffix_only_adler32": suffix_only_adler32,
            "radix_prefix_hit_causes_false_mismatch": radix_prefix_hit_causes_false_mismatch,
            "prefill_aux_buffers": prefill_aux_buffers,
            "decode_aux_buffers": decode_aux_buffers,
            "transferred_aux_buffers": transferred_aux_buffers,
        },
        "sglang_ltx2_vec_qknorm_rope": {
            "num_heads": ltx2_num_heads,
            "head_dim": ltx2_head_dim,
            "num_pairs_per_token": ltx2_num_pairs,
            "vec_width_pairs": ltx2_vec_width_pairs,
            "num_vecs_per_token": ltx2_num_vecs_per_token,
            "tx_reduction_factor": ltx2_tx_reduction_factor,
        },
        "sglang_qsa_prefill_gather": {
            "compress_ratio": qsa_compress_ratio,
            "seq_lengths": qsa_seq_lengths,
            "complete_blocks_per_seq": qsa_complete_blocks_per_seq,
            "total_complete_blocks": qsa_total_complete_blocks,
            "legacy_launches": qsa_legacy_launches,
            "new_triton_launches": qsa_new_triton_launches,
        },
        "sglang_eagle_hicache_restore": {
            "decode_prefix_bigrams": eagle_decode_prefix_bigrams,
            "boundary_tokens": eagle_boundary_tokens,
            "rematch_token_len": eagle_rematch_token_len,
        },
        "vllm_rocm_triton_ptr_range": {
            "rocm_32bit_limit_bytes": rocm_32bit_limit_bytes,
            "total_storage_bytes": total_storage_bytes,
            "layer_region_bytes": layer_region_bytes,
            "key_view_ptr_range_bytes": key_view_ptr_range_bytes,
            "value_view_ptr_range_bytes": value_view_ptr_range_bytes,
            "uses_32bit_buffer_ops_before": rocm_uses_32bit_buffer_ops_before,
            "uses_32bit_buffer_ops_after": rocm_uses_32bit_buffer_ops_after,
        },
        "vllm_mrv2_dbo": {
            "ubatch_token_offsets": dbo_ubatch_token_offsets,
            "captured_rows_before": dbo_captured_rows_before,
            "captured_rows_after": dbo_captured_rows_after,
            "reported_ubatch_before": dbo_reported_ubatch_before,
            "reported_ubatch_after": dbo_reported_ubatch_after,
        },
        "vllm_chunked_local_cpu_offload": {
            "attention_chunk_size": attention_chunk_size,
            "block_size": block_size,
            "cases": evaluated_cases,
        },
        "trtllm_minimax_m3_vl_and_lora": {
            "bare_image_token": minimax_bare_image_token,
            "bare_video_token": minimax_bare_video_token,
            "chat_template_applied_flag": minimax_chat_template_applied_flag,
            "lora_reloaded_ckpt_source_before": lora_reloaded_ckpt_source_before,
            "lora_reloaded_ckpt_source_after": lora_reloaded_ckpt_source_after,
        },
    }


def validate_summary(summary: dict[str, Any]) -> None:
    assert summary["sglang_pd_kv_checksum"]["radix_prefix_hit_causes_false_mismatch"] is True
    assert summary["sglang_pd_kv_checksum"]["transferred_aux_buffers"] == 2
    assert summary["sglang_ltx2_vec_qknorm_rope"]["tx_reduction_factor"] == 8
    assert summary["sglang_qsa_prefill_gather"]["total_complete_blocks"] == 1137
    assert summary["sglang_qsa_prefill_gather"]["legacy_launches"] == 5
    assert summary["sglang_qsa_prefill_gather"]["new_triton_launches"] == 1
    assert summary["sglang_eagle_hicache_restore"]["rematch_token_len"] == 65
    assert summary["vllm_rocm_triton_ptr_range"]["uses_32bit_buffer_ops_before"] is False
    assert summary["vllm_rocm_triton_ptr_range"]["uses_32bit_buffer_ops_after"] is True
    assert summary["vllm_mrv2_dbo"]["captured_rows_after"] == 128
    assert summary["vllm_mrv2_dbo"]["reported_ubatch_after"] == 1
    assert len(summary["vllm_chunked_local_cpu_offload"]["cases"]) == 6
    assert summary["trtllm_minimax_m3_vl_and_lora"]["lora_reloaded_ckpt_source_after"] == "nemo"


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
