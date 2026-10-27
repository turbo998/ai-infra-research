"""Phase 112 Track B deterministic offline model:
SGLang Paged-Experts Single-Sync Residency & Diffusion Plucker/MoE Fixes,
vLLM Bounded Prompt-Score Prefix Reuse, DSv4.1 FP8 Sparse MQA Indexer, Mooncake Vectorized Runs & Process Checkpoint API,
and TensorRT-LLM Rubin SM107 MiniMax-M3 P128 FMHA & Low-M GEMM Capture Safety.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def simulate_sglang_paged_experts_and_diffusion() -> dict[str, object]:
    """Model sglang@a4a19332ab (paged-experts single D2H sync),
    sglang@69767e5aa6 (float32 Plucker rays for bf16 poses),
    and sglang@c45ebb507d (HunyuanImage-3 in-place MoE shared MLP & uniform Euler sigmas).
    """
    # 1. Paged-experts D2H sync count per host-planned layer step:
    # Legacy: torch.unique(topk_ids).tolist() (1 sync) + slot_expert.tolist() (1 sync)
    # + slot_lastuse.tolist() (1 sync) + step.item() in store_from (1 sync) = 4 D2H syncs/layer
    # Fixed: torch.cat([slot_expert, slot_lastuse, step, counts]).tolist() = 1 D2H sync/layer
    layers = 61
    sync_cost_us = 14.0
    legacy_syncs_per_layer = 4
    fixed_syncs_per_layer = 1
    legacy_step_sync_us = layers * legacy_syncs_per_layer * sync_cost_us
    fixed_step_sync_us = layers * fixed_syncs_per_layer * sync_cost_us

    # 2. BF16 vs FP32 pixel coordinate representation for Plucker rays (width=640)
    def bf16_round_int(x: int) -> float:
        if x <= 256:
            return float(x)
        if x <= 512:
            return float((x // 2) * 2)
        return float((x // 4) * 4)

    width = 640
    bf16_distinct_cols = len({bf16_round_int(col) for col in range(width)})
    fp32_distinct_cols = len({float(col) for col in range(width)})

    # 3. HunyuanImage-3 in-place FusedMoE overwriting input before shared_mlp
    x_in = [1.5, -2.0, 0.5, 3.0]
    # Buggy: experts overwrite x_in with 10*x_in in place, then shared_mlp sees 10*x_in -> total 20*x_in
    buggy_out = [10.0 * v + 10.0 * v for v in x_in]
    # Fixed: shared_mlp sees original layer input x_in -> total 11*x_in
    fixed_out = [10.0 * v + 1.0 * v for v in x_in]

    return {
        "legacy_d2h_syncs_per_step": layers * legacy_syncs_per_layer,
        "fixed_d2h_syncs_per_step": layers * fixed_syncs_per_layer,
        "d2h_sync_reduction_ratio": legacy_syncs_per_layer / fixed_syncs_per_layer,
        "saved_sync_overhead_us_per_step": round(legacy_step_sync_us - fixed_step_sync_us, 2),
        "bf16_distinct_pixel_columns_640": bf16_distinct_cols,
        "fp32_distinct_pixel_columns_640": fp32_distinct_cols,
        "collapsed_columns_in_bf16": width - bf16_distinct_cols,
        "hunyuan_buggy_scale_factor": round(buggy_out[0] / x_in[0], 2),
        "hunyuan_fixed_scale_factor": round(fixed_out[0] / x_in[0], 2),
    }


def simulate_vllm_prefix_score_ds41_fp8_mooncake_and_checkpoint() -> dict[str, object]:
    """Model vllm@e1e77a279d (prompt_logprob_start bounded prefix cache reuse),
    vllm@a2ad4b74b9 (DSv4.1 DeepGEMM sparse MQA logits on FP8 indexer cache),
    vllm@2897401370 (Mooncake vectorized _contiguous_runs),
    vllm@22d1bc8814 (Rust frontend final-chunk kv_transfer_params),
    and vllm@05a303fdc8 (HTTP process checkpoint lifecycle).
    """
    # 1. Prefix cache reuse bounded by prompt_logprob_start (vllm@e1e77a279d)
    block_size = 4
    prompt_len = 16
    cached_tokens = 16

    def compute_reusable_prefix_tokens(
        start: int | None, skip_reading_prefix_cache: bool, for_external_connector: bool
    ) -> int:
        if skip_reading_prefix_cache or for_external_connector or start is None or start <= 0:
            return 0
        # Rows from start+1..prompt_len-1 are scored; we can reuse full blocks up to start
        return min(cached_tokens, (start // block_size) * block_size)

    hit_start0_local = compute_reusable_prefix_tokens(0, False, False)
    hit_start5_local = compute_reusable_prefix_tokens(5, False, False)
    hit_start5_external = compute_reusable_prefix_tokens(5, False, True)

    # 2. DSv4.1 Sparse MQA Logits on SM100 FP8 Indexer Cache (vllm@a2ad4b74b9)
    # Power-of-two scale per row stored as 4 equal UE8M0 bytes packed into int32
    row_amax = 14.0
    exp_val = math.ceil(math.log2(max(row_amax, 1e-4) / 448.0))
    ue8m0_byte = (exp_val + 127) & 0xFF
    packed_ue8m0_int32 = (
        ue8m0_byte | (ue8m0_byte << 8) | (ue8m0_byte << 16) | (ue8m0_byte << 24)
    )
    context_tokens = 65536
    candidate_blocks = 64
    block_tokens = 64
    dense_scored_tokens = context_tokens
    sparse_scored_tokens = candidate_blocks * block_tokens

    # 3. Mooncake vectorized _contiguous_runs (vllm@2897401370)
    src_ids = [10, 11, 12, 20, 21, 30, 31, 32]
    dst_ids = [100, 101, 102, 200, 201, 305, 306, 307]
    breaks = [
        i + 1
        for i in range(len(src_ids) - 1)
        if (src_ids[i + 1] - src_ids[i] != 1) or (dst_ids[i + 1] - dst_ids[i] != 1)
    ]
    starts = [0] + breaks
    ends = starts[1:] + [len(src_ids)]
    lengths = [e - s for s, e in zip(starts, ends)]

    # 4. Process checkpoint HTTP status codes (vllm@05a303fdc8)
    status_http_codes = {
        "running": 503,
        "preparing": 503,
        "prepared": 200,
        "restoring": 503,
        "failed": 500,
    }

    return {
        "prefix_hit_start0_local": hit_start0_local,
        "prefix_hit_start5_local": hit_start5_local,
        "prefix_hit_start5_external_connector": hit_start5_external,
        "fp8_indexer_ue8m0_byte": ue8m0_byte,
        "fp8_indexer_packed_ue8m0_hex": hex(packed_ue8m0_int32),
        "ds41_64k_logit_work_reduction_x": dense_scored_tokens / sparse_scored_tokens,
        "mooncake_contiguous_starts": starts,
        "mooncake_contiguous_lengths": lengths,
        "checkpoint_status_http_codes": status_http_codes,
        "rust_final_chunk_carries_kv_transfer_params": True,
    }


def simulate_trtllm_sm107_minimax_m3_and_low_m_gemm() -> dict[str, object]:
    """Model TensorRT-LLM@854ccee8a9 (MiniMax-M3 on Rubin SM107: NVFP4 KV exemption + 128-token FMHA)
    and TensorRT-LLM@b0af321b0c (Keep low-M GEMM outputs distinct during CUDA graph capture).
    """
    # 1. SM107 MiniMax-M3 layer routing & KV dtype assignment
    sm107_routes = {
        "sparse_target_layers": {"kv_dtype": "NVFP4", "backend": "MSA_Triton"},
        "dense_target_layers": {"kv_dtype": "FP8", "backend": "SDPA"},
        "eagle3_shared_draft_layer_p128": {
            "kv_dtype": "FP8",
            "tokens_per_block": 128,
            "backend": "FlashInferTrtllmGenFmha",
        },
    }

    # 2. Low-M GEMM output buffer allocation during CUDA graph capture
    # Suppose lm_head (same module_key, m_bucket=16, n=151936) runs 3 times in a captured draft loop.
    # Legacy cached a single buffer in self._output_buffers[buf_key] if pre-warmed, so all 3 calls
    # aliased the same tensor address and overwrote draft step 0 & 1 logits with step 2 logits!
    step_values = [11.0, 22.0, 33.0]
    shared_buf = [0.0]
    legacy_retained_views = []
    for v in step_values:
        shared_buf[0] = v
        legacy_retained_views.append(shared_buf)  # aliased reference
    legacy_readback = [view[0] for view in legacy_retained_views]

    fixed_retained_views = []
    for v in step_values:
        distinct_buf = [v]  # torch.empty per call when is_capturing=True
        fixed_retained_views.append(distinct_buf)
    fixed_readback = [view[0] for view in fixed_retained_views]

    return {
        "sm107_layer_routes": sm107_routes,
        "legacy_captured_logits_aliased": legacy_readback == [33.0, 33.0, 33.0],
        "fixed_captured_logits_distinct": fixed_readback == step_values,
        "legacy_readback": legacy_readback,
        "fixed_readback": fixed_readback,
    }


def run_all() -> dict[str, object]:
    sglang_res = simulate_sglang_paged_experts_and_diffusion()
    vllm_res = simulate_vllm_prefix_score_ds41_fp8_mooncake_and_checkpoint()
    trtllm_res = simulate_trtllm_sm107_minimax_m3_and_low_m_gemm()

    assert sglang_res["d2h_sync_reduction_ratio"] == 4.0
    assert sglang_res["collapsed_columns_in_bf16"] == 224
    assert sglang_res["hunyuan_fixed_scale_factor"] == 11.0
    assert vllm_res["prefix_hit_start0_local"] == 0
    assert vllm_res["prefix_hit_start5_local"] == 4
    assert vllm_res["prefix_hit_start5_external_connector"] == 0
    assert vllm_res["ds41_64k_logit_work_reduction_x"] == 16.0
    assert vllm_res["mooncake_contiguous_starts"] == [0, 3, 5]
    assert vllm_res["mooncake_contiguous_lengths"] == [3, 2, 3]
    assert trtllm_res["legacy_captured_logits_aliased"]
    assert trtllm_res["fixed_captured_logits_distinct"]

    return {
        "phase": 112,
        "track": "B",
        "title": "SGLang Paged-Experts Single-Sync, vLLM Bounded Prefix-Score & DSv4.1 FP8 Sparse Indexer, TRT-LLM SM107 MiniMax-M3",
        "sglang_paged_experts_and_diffusion": sglang_res,
        "vllm_prefix_ds41_mooncake_checkpoint": vllm_res,
        "trtllm_sm107_and_low_m_gemm": trtllm_res,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_all()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        print("Phase 112 Track B check passed.")


if __name__ == "__main__":
    main()
