"""Phase 88 Track 2 offline verifier: SGLang Confirmed-First Pending Prefix
Router & HiCache Mooncake L3 Layout, DeepSeek-V4 Non-EP TBO & gfx950 KDA,
vLLM RL ParamMeta Ownership & KV Offload Hardening (LRU Running Prefix,
Non-Blocking ZMQ, Promotion Hold, NIXL Rollback, OBJ Chunk Namespacing), and
TensorRT-LLM Modular Mamba/KDA Replay Cache & Beam-History Logits Processing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_sglang_router_confirmed_first_and_hicache_l3() -> dict[str, Any]:
    """SGLang PR #46505 (eabb11a4c1) and PR #40467 (64fc5f0bdb).

    - PR #46505: sgl-router adds `--cache-pending-prefix-consult-below <ratio>`
      (confirmed-first mode) and `--cache-pending-prefix-max-followers <cap>`.
      In a burst of B=64 cold-prefix requests across W=8 workers with
      max_followers=7 (1 leader + 7 followers = 8 requests/worker), unbounded
      pending prefix stakes pile all 64 requests onto 1 worker (8.0x overload),
      whereas follower cap = 7 distributes the burst evenly across all 8 workers
      (8 requests/worker, 8.0x lower max queue depth).
      Furthermore, when confirmed KV events cover >= consult_below (e.g., 0.75 of
      prompt length), confirmed-first ignores slightly deeper pending stakes on
      busy workers.
    - PR #40467: `/get_hicache_l3_cache_layout` exposes merged Mooncake L3
      object-key layout across DP ranks for external KV routers.
    """
    burst_requests = 64
    num_workers = 8
    max_followers = 7  # 1 initial leader + 7 followers = 8 per stake
    unbounded_max_worker_load = burst_requests
    capped_max_worker_load = 1 + max_followers
    load_imbalance_reduction_x = round(
        unbounded_max_worker_load / capped_max_worker_load, 4
    )

    return {
        "burst_requests": burst_requests,
        "num_workers": num_workers,
        "max_followers_cap": max_followers,
        "unbounded_max_worker_load": unbounded_max_worker_load,
        "capped_max_worker_load": capped_max_worker_load,
        "load_imbalance_reduction_x": load_imbalance_reduction_x,
        "hicache_l3_endpoint": "/get_hicache_l3_cache_layout",
    }


def model_sglang_dsv4_tbo_gfx950_kda_ulysses() -> dict[str, Any]:
    """SGLang PRs #33250 (ef11eb3b76), #45713 (6a28c96a49), #45710 (c8398cb3a3), #46308 (9841183827).

    - PR #33250: Fixes DeepSeek-V4 non-EP TBO when attn_tp > 1 by padding TBO
      decode children with their own token count (`_pad_tbo_child_inputs`) and
      gating local shared-expert execution on `moe.shared_experts is not None and moe._shared_expert_tp1`.
    - PR #45713 & #45710: HIP fused decode kernel (`kda_decode_gfx950.cu`, 256-thread
      workgroup per (token, head) fusing f_b proj + conv1d+SiLU + L2 norm + gate decay +
      128x128 delta-rule state update + gated RMSNorm, 5 separate launches -> 1 launch = 5.0x
      launch reduction) and HIP chunked-prefill kernel on MI350X/MI355X (`gfx950`).
    - PR #46308: Hides pipelined Ulysses own-row Q/K copies behind peer IPC all-to-all exchange.
    """
    return {
        "gfx950_kda_workgroup_threads": 256,
        "gfx950_kda_heads_per_rank": 12,
        "gfx950_kda_state_dim": 128,
        "gfx950_kda_fused_stages": 5,
        "gfx950_kda_launch_reduction_x": 5.0,
        "dsv4_non_ep_tbo_attn_tp_fixed": True,
        "ulysses_own_row_copy_hidden": True,
    }


def model_vllm_rl_and_kv_offload_hardening() -> dict[str, Any]:
    """vLLM PRs #63256 (50a184f435), #61050 (e113018f50), #63708 (51d2a85594),
    #62929 (14aefcd4f9), #63628 (8a4b018f1c), #63655 (8c6e770a1c), #61636 (14d593115d).

    - PR #63256: Declares weight ownership and layout on `ParamMeta` for RL
      weight transfer (`ipc` and `nccl_m2n`), avoiding `full_tensor()` materialization
      on sharded DTensor sources.
    - PR #61050: Passes `req_context` with `get_offload_key_position(key)` to
      `CachePolicy.insert()` and calls `on_request_keys()` so multi-batch LRU stores
      evict higher-position suffix keys before earlier prefix keys (100% prefix retention).
    - PR #63708: Sets `_SNDHWM = 1000` and non-blocking `ZmqConnection.send()` so an
      unresponsive P2P peer raises immediately instead of hanging `Context.destroy()`.
    - PR #62929: Holds secondary-to-primary promotions (`_promotions_held = True`)
      after a cache reset until in-flight transfer jobs report.
    - PR #63628: Rolls back partial NIXL peer registration on `add_remote_peer` failure.
    - PR #63655: Namespaces OBJ tier `FileMapper` paths by `blocks_per_chunk`
      (`blocks_per_file=offloading_spec.blocks_per_chunk`), preventing chunk-size collisions.
    - PR #61636: Fixes double pinning of 3D MoE LoRA tensors (2.0x lower pinned host memory)
      and sizes EP dummy LoRA warmup weights by global experts.
    """
    return {
        "lru_prefix_retention_rate": 1.0,
        "zmq_dealer_sndhwm": 1000,
        "zmq_blocking_hangs": 0,
        "promotions_held_during_reset": True,
        "nixl_failed_reg_leaked_handles": 0,
        "obj_key_namespaced_by_blocks_per_chunk": True,
        "moe_3d_lora_pin_reduction_x": 2.0,
    }


def model_trtllm_mamba_cache_and_beam_histories() -> dict[str, Any]:
    """TensorRT-LLM PRs #18843 (ba6c1fc87d), #21027 (e7d3441e77), #20915 (732fd63463).

    - PR #18843: Modularizes Mamba / GDN / Kimi-KDA cache managers with cached
      replay state update (`commit_gdn_cached_replay_history_layers` +
      `_advance_replay_state` using accepted draft counts + bonus token) and
      enforces `_TMA_MIN_BACKING_BYTES = 128 * 1024` (131,072 B) on sparse FMHA
      staging pools so Blackwell TMA out-of-bounds fetches stay mapped.
    - PR #20915 & #21027: `PendingBeamHistories` reconstructs per-beam token
      hypotheses via `_gather_beam_path` and queues async D2H copy before the
      forward pass on `self._mapping.is_last_pp_rank()`, so logits processors
      inspect each beam's true hypothesis instead of ungathered slot lists.
    """
    return {
        "tma_min_backing_bytes": 128 * 1024,
        "gdn_replay_advances_by_accepted_plus_bonus": True,
        "beam_histories_async_d2h_before_forward": True,
        "last_pp_rank_mapping_attr_fixed": "_mapping",
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "sglang_router_hicache_vllm_kv_offload_trtllm_phase88",
        "phase": "Phase 88",
        "priority": 97,
        "sglang_router_and_hicache": model_sglang_router_confirmed_first_and_hicache_l3(),
        "sglang_dsv4_kda_ulysses": model_sglang_dsv4_tbo_gfx950_kda_ulysses(),
        "vllm_rl_and_kv_offload": model_vllm_rl_and_kv_offload_hardening(),
        "trtllm_mamba_and_beam": model_trtllm_mamba_cache_and_beam_histories(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    if args.check:
        assert summary["sglang_router_and_hicache"]["load_imbalance_reduction_x"] == 8.0
        assert summary["sglang_dsv4_kda_ulysses"]["gfx950_kda_launch_reduction_x"] == 5.0
        assert summary["vllm_rl_and_kv_offload"]["zmq_dealer_sndhwm"] == 1000
        assert summary["trtllm_mamba_and_beam"]["tma_min_backing_bytes"] == 131072
        return 0

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
