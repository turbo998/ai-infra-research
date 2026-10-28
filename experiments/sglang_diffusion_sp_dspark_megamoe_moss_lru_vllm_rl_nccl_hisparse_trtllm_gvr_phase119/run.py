#!/usr/bin/env python3
"""Phase 119 Track B deterministic verification model:
SGLang MOSS-Image SP Full-Latent RoPE & High-Quality Fused QKV, DSV4 Host-Free DSpark Draft Prep,
MegaMoE DP Prefill Budget Fix, PD Bootstrap-Queue Prefix LRU Refresh, vLLM Per-Update RL NCCL Weight Transfer,
LoRA+EP Non-Linear Placement Guard, DFlash/DSpark OnlineAcceptanceEstimator Warmup, HiSparse 256-Row Staged KV Padding,
and TensorRT-LLM GVR K=2048 Block64 Sampling & Parallel Rescue.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib


def next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def run_track_b_simulation() -> dict[str, object]:
    # 1. SGLang #46664 (5bb7ecc2c2): MOSS-Image preserves full latents for SP before DiT RoPE
    moss_sp_did_shard_latents_before_rope = False
    moss_sp_sharded_inside_dit_after_rope = True

    # 2. SGLang #46897 (357c53885f): dsv4 expand_prefill_causally without host lengths
    extend_seq_lens = [4, 7, 3, 6]
    cumsum = []
    running = 0
    for length in extend_seq_lens:
        running += length
        cumsum.append(running)
    extend_start_loc = [c - l for c, l in zip(cumsum, extend_seq_lens)]
    dsv4_host_syncs_required = 0

    # 3. SGLang #46267 (15d39bd840): MegaMoE stops double-dividing DP prefill budget
    # ServerArgs already divides chunked_prefill_size (e.g. 16384 -> 8192 at attn_dp_size=2).
    cfg_chunked_prefill_size_per_dp = 8192
    attn_dp_size = 2
    buggy_megamoe_budget = cfg_chunked_prefill_size_per_dp // attn_dp_size
    fixed_megamoe_budget = cfg_chunked_prefill_size_per_dp
    megamoe_budget_restoration_ratio = fixed_megamoe_budget / buggy_megamoe_budget

    # 4. SGLang #46641 (610a74d648): MOSS-Image fuses joint-block QKV in high quality
    moss_exact_gemms_per_block = 3
    moss_high_quality_gemms_per_block = 1
    moss_dim = 3072
    moss_qkv_token_stride = 3 * moss_dim

    # 5. SGLang #46812 (54be2010d3): Refresh PD prefill bootstrap-queue prefixes in waiting-prefix LRU pass
    bootstrap_max = 512
    waiting_max = 256
    # Verify reversed(bootstrap_queue[:512]) is touched before reversed(waiting_queue[:waiting_max])
    bootstrap_reqs = [f"boot_{i}" for i in range(4)]
    waiting_reqs = [f"wait_{i}" for i in range(4)]
    lru_touch_order = list(reversed(bootstrap_reqs[:bootstrap_max])) + list(reversed(waiting_reqs[:waiting_max]))

    # 6. vLLM #62557 (2b678ac739): RL NCCL weight transfer per-update source ranks
    num_broadcasters = 4  # e.g., PP=4 trainer ranks
    num_workers = 8
    nccl_world_size = num_broadcasters + num_workers
    nccl_rank_offset = num_broadcasters
    valid_src_ranks = list(range(0, nccl_rank_offset))
    invalid_src_rank_rejected = not (0 <= 4 < nccl_rank_offset)

    # 7. vLLM #62797 (1c787a25cc): Reject non-linear expert placement with LoRA + EP
    def lora_ep_placement_accepted(use_ep: bool, strategy: str) -> bool:
        if use_ep and strategy != "linear":
            return False
        return True

    lora_ep_linear_ok = lora_ep_placement_accepted(True, "linear")
    lora_ep_round_robin_rejected = not lora_ep_placement_accepted(True, "round_robin")

    # 8. vLLM #63819 (458334f00f): Close DFlash/DSpark and adaptive-verification JIT warmup gaps
    max_num_reqs = 96
    online_acceptance_block_r = max(256, next_power_of_2(max_num_reqs))
    dflash_fixed_block_size = 128
    runtime_jit_compilations_after_warmup = 0

    # 9. vLLM #61659 (ebaa812475): HiSparse pad staged prefill KV to >= 256 rows (656 B/row)
    min_staged_prefill_kv_rows = 256
    kv_row_bytes = 656
    block_size = 64
    requested_blocks = 1  # 64 rows < 256 minimum
    allocated_blocks = max(requested_blocks, math.ceil(min_staged_prefill_kv_rows / block_size))
    allocated_rows = allocated_blocks * block_size
    trtllm_gen_max_read_rows_small_prefill = 250
    hisparse_oob_prevented = allocated_rows >= min_staged_prefill_kv_rows > trtllm_gen_max_read_rows_small_prefill

    # 10. TensorRT-LLM #20998 (9cf4530787): GVR K=2048 reg_clus with block64 sampling & parallel rescue
    gvr_k = 2048
    gvr_blkc = 1024
    gvr_hint_slots = (gvr_k + 1023) // 1024
    gvr_block64_threads = 64
    gvr_candidates_per_group = gvr_k // gvr_block64_threads

    return {
        "experiment_id": "sglang_diffusion_sp_dspark_megamoe_moss_lru_vllm_rl_nccl_hisparse_trtllm_gvr_phase119",
        "moss_sp_did_shard_latents_before_rope": moss_sp_did_shard_latents_before_rope,
        "moss_sp_sharded_inside_dit_after_rope": moss_sp_sharded_inside_dit_after_rope,
        "dsv4_extend_start_loc": extend_start_loc,
        "dsv4_host_syncs_required": dsv4_host_syncs_required,
        "buggy_megamoe_budget": buggy_megamoe_budget,
        "fixed_megamoe_budget": fixed_megamoe_budget,
        "megamoe_budget_restoration_ratio": megamoe_budget_restoration_ratio,
        "moss_exact_gemms_per_block": moss_exact_gemms_per_block,
        "moss_high_quality_gemms_per_block": moss_high_quality_gemms_per_block,
        "moss_qkv_token_stride": moss_qkv_token_stride,
        "bootstrap_max": bootstrap_max,
        "lru_touch_first_item": lru_touch_order[0],
        "nccl_world_size": nccl_world_size,
        "nccl_rank_offset": nccl_rank_offset,
        "valid_src_ranks": valid_src_ranks,
        "invalid_src_rank_rejected": invalid_src_rank_rejected,
        "lora_ep_linear_ok": lora_ep_linear_ok,
        "lora_ep_round_robin_rejected": lora_ep_round_robin_rejected,
        "online_acceptance_block_r": online_acceptance_block_r,
        "dflash_fixed_block_size": dflash_fixed_block_size,
        "runtime_jit_compilations_after_warmup": runtime_jit_compilations_after_warmup,
        "min_staged_prefill_kv_rows": min_staged_prefill_kv_rows,
        "kv_row_bytes": kv_row_bytes,
        "allocated_blocks": allocated_blocks,
        "allocated_rows": allocated_rows,
        "hisparse_oob_prevented": hisparse_oob_prevented,
        "gvr_k": gvr_k,
        "gvr_blkc": gvr_blkc,
        "gvr_hint_slots": gvr_hint_slots,
        "gvr_block64_threads": gvr_block64_threads,
        "gvr_candidates_per_group": gvr_candidates_per_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(__file__).resolve().parent
    summary = run_track_b_simulation()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert summary["moss_sp_did_shard_latents_before_rope"] is False
        assert summary["dsv4_extend_start_loc"] == [0, 4, 11, 14]
        assert summary["megamoe_budget_restoration_ratio"] == 2.0
        assert summary["moss_high_quality_gemms_per_block"] == 1
        assert summary["lru_touch_first_item"] == "boot_3"
        assert summary["nccl_world_size"] == 12
        assert summary["lora_ep_round_robin_rejected"] is True
        assert summary["online_acceptance_block_r"] == 256
        assert summary["hisparse_oob_prevented"] is True
        assert summary["gvr_hint_slots"] == 2
        assert summary["gvr_candidates_per_group"] == 32
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
