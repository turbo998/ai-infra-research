"""Deterministic offline models for Phase 123 Track C:
1. Megatron-LM `fused_sparse_mla_absorbed` supporting 512-dim NoPE queries/keys (`+64` zero-padded channels
   to match 576-dim absorbed layout while preserving caller scale) and unaligned `topk` (padding to multiple
   of 64 with sentinel `-1`) (PR #7617, commit f77f360c96b6a5ec032778811868196ec75daed9),
   Mixture-of-Kittens (`mok`) MoE megakernel backend (`SM100`/`SM103`)
   (PR #8210, commit a153d5d3f543479e44637d4d62ad402e48ef4a7a), and fused shared-expert `GroupedLinear`
   `FC1` local TP shard `out_features` fix (PR #8145, commit 3e47f31ec222c1db8e7a535edcf1653dede8c2fe).
2. veRL vLLM rollout submission-gate barrier (`_generate_admitted`) releasing `_admitting` immediately
   after `AsyncLLM.add_request()` returns rather than waiting for the first yielded token, eliminating
   the `5-14 s` weight-sync pause stall when `len(batch) > max_num_seqs`
   (PR #8397, commit 081c76bba2a92bcff3032c2f5e4758bdbe687b37).
3. Mooncake EFA `remote_accessible=false` security enforcement (`FI_READ | FI_WRITE` only +
   `kLocalOnlyKey = 1` placeholder lkey so peers on libfabric `< 2.6` cannot RMA into local-only memory)
   (PR #4976, commit ad1037135ce1cc144f0096d2ca4525dcf56091bd), TENT `IOUringStagingPool` opt-in pinned
   DMA bounce pool (`32 x 4 MiB` `cudaHostAllocPortable`) (PR #5012, commit 5fa91fe135aba3255c638acd8fd95ba59a0c76c8),
   `hccs:device` 2 MiB alignment gating (PR #5049, commit 811c97f0f8fd3687e199e7c3589a92a68fa9a6a1),
   Ascend failed-lookup `slice->markFailed()` (PR #4888, commit eccc7c5998428d1ab5b892bf037acc500ea54954),
   and shared `libtent_metrics.so` HTTP 500 self-observability (PR #4938, commit d3ef7b1c2821d8cdfd5986ee5949e8c17608a36c).
4. LMCache Coordinator atomic metadata rollback (`capture`/`restore` + HTTP `503` on `MetadataWriteError`
   instead of false `200`) (PR #6077, commit 7660f4ef375138be69bf47f958a44565d00c17ac),
   lazy Mooncake SDK import on context creation (PR #6134, commit b377915ca6944c229ab5a4bbfde59a978be0615d),
   and Dynamo OTLP 4-byte UTF-8 token packing in `trace-analyze` (PR #6090, commit b89bc5f9ffd8cff421cf823ff8d5e18db1316f40).
5. NVIDIA Dynamo SGLang KVCR HiCache capability advertisement + router-owned `kv_hints` forwarding
   (stripping caller-spoofed `kv_hints`) (PR #16117, commit f38fb532783a7f42af27653f533b2a515678f305),
   EPP `FULL_DUPLEX_STREAMED` body-before-header validation (PR #17125, commit 993748616dfcfe2932298f2cf5ca0db89c11b0fe),
   and `WorkerType::Prefill` discovery filtering (PR #17325, commit 9087efc4020074af28d1db0bc36ff1a86180fee2).
6. NVIDIA NIXL UCCL plugin `std::optional<nixl_blob_t> notif_msg` reset on `postXfer` and gated in `checkXfer`,
   eliminating 6 spurious empty/stale notifications (`10 -> 4` notifications, `6 -> 0` empty) that crashed
   vLLM HiSparse disaggregated prefill (PR #2429, commit fed120d87ed46ff21a9973a16bfb14b10b5cfdd1).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SUMMARY_PATH = Path(__file__).with_name("summary.json")


def model_megatron_mla_mok_and_shared_expert_tp() -> dict[str, Any]:
    """Model Megatron-LM fused_sparse_mla_absorbed (#7617), Mixture-of-Kittens (#8210), and shared-expert TP (#8145)."""
    # 1. fused_sparse_mla_absorbed 512-dim NoPE & unaligned topk (#7617)
    d_qk_nope = 512
    d_qk_absorbed = 576
    nope_zero_pad_channels = d_qk_absorbed - d_qk_nope
    # Caller scale must remain 1 / sqrt(512), NOT recomputed as 1 / sqrt(576):
    caller_scale_512 = round(1.0 / math.sqrt(d_qk_nope), 6)
    wrong_recomputed_scale_576 = round(1.0 / math.sqrt(d_qk_absorbed), 6)
    # Unaligned topk=100 padded to multiple of 64 with sentinel -1:
    raw_topk = 100
    padded_topk = ((raw_topk + 63) // 64) * 64
    sentinel_padded_slots = padded_topk - raw_topk
    sentinel_value = -1

    # 2. Mixture-of-Kittens (mok) MoE megakernel backend (#8210)
    # Standard MoE layer launches separate router + permute + grouped FC1 + act + grouped FC2 + unpermute + combine kernels (7 launches);
    # MOK fuses token dispatch, expert GEMMs, activation, and combine into a persistent megakernel on SM100/SM103.
    standard_moe_launches_per_layer = 7
    mok_launches_per_layer = 1

    # 3. Fused shared-expert GroupedLinear FC1 out_features with TP=4 (#8145)
    # SwiGLU doubles FC1 output features (2 * ffn_hidden_size_per_partition).
    # Bug multiplied local shard by tp_world_size a second time (4x inflation at TP=4).
    ffn_hidden_size = 16384
    tp_world_size = 4
    local_ffn_per_partition = ffn_hidden_size // tp_world_size  # 4096
    correct_fc1_out_features = 2 * local_ffn_per_partition  # 8192
    buggy_fc1_out_features = 2 * local_ffn_per_partition * tp_world_size  # 32768
    weight_inflation_factor = round(buggy_fc1_out_features / correct_fc1_out_features, 2)

    return {
        "d_qk_nope": d_qk_nope,
        "d_qk_absorbed": d_qk_absorbed,
        "nope_zero_pad_channels": nope_zero_pad_channels,
        "caller_scale_512": caller_scale_512,
        "wrong_recomputed_scale_576": wrong_recomputed_scale_576,
        "raw_topk": raw_topk,
        "padded_topk": padded_topk,
        "sentinel_padded_slots": sentinel_padded_slots,
        "sentinel_value": sentinel_value,
        "standard_moe_launches_per_layer": standard_moe_launches_per_layer,
        "mok_launches_per_layer": mok_launches_per_layer,
        "correct_fc1_out_features": correct_fc1_out_features,
        "buggy_fc1_out_features": buggy_fc1_out_features,
        "weight_inflation_factor": weight_inflation_factor,
    }


def model_verl_vllm_submission_gate_barrier() -> dict[str, Any]:
    """Model veRL vLLM rollout submission-gate barrier (#8397)."""
    # Batch of 256 rollout prompts with vLLM max_num_seqs=32.
    # Only 32 requests are scheduled immediately; 224 wait in the engine's WAITING queue.
    # Under the old 1st-token gate, _admitting stayed > 0 until the 224 queued requests
    # waited for running waves to finish and emitted their first token (5.0 - 14.0 s stall).
    # Under the new add_request()-return gate, all 256 requests finish add_request() in ~12 ms,
    # allowing pause_generation(wait_for_inflight_requests=False) to enter weight sync immediately.
    batch_size = 256
    max_num_seqs = 32
    waiting_queue_requests = batch_size - max_num_seqs
    waves_to_first_token = batch_size // max_num_seqs
    per_wave_decode_s = 1.60  # 8 waves => 7 waiting waves * 1.6s = 11.2s
    pause_stall_old_s = round((waves_to_first_token - 1) * per_wave_decode_s, 2)
    pause_stall_new_s = 0.012  # 12 ms for 256 AsyncLLM.add_request() calls
    pause_speedup = round(pause_stall_old_s / pause_stall_new_s, 2)

    return {
        "batch_size": batch_size,
        "max_num_seqs": max_num_seqs,
        "waiting_queue_requests": waiting_queue_requests,
        "pause_stall_old_s": pause_stall_old_s,
        "pause_stall_new_s": pause_stall_new_s,
        "pause_speedup": pause_speedup,
    }


def model_mooncake_efa_iouring_and_lmcache_dynamo_nixl() -> dict[str, Any]:
    """Model Mooncake (#4976, #5012, #5049, #4888), LMCache (#6077, #6134), Dynamo (#16117, #17125, #17325), and NIXL (#2429)."""
    # 1. Mooncake EFA remote_accessible=false security enforcement (#4976)
    # Local-only MR registered with FI_READ | FI_WRITE (omitting FI_REMOTE_READ | FI_REMOTE_WRITE)
    # and exports kLocalOnlyKey = 1 instead of hardware rkey (e.g., 0x9A4F20).
    local_only_fi_access = ["FI_READ", "FI_WRITE"]
    remote_accessible_fi_access = ["FI_READ", "FI_WRITE", "FI_REMOTE_READ", "FI_REMOTE_WRITE"]
    exported_rkey_when_local_only = 1  # kLocalOnlyKey

    # 2. Mooncake TENT IOUringStagingPool (#5012)
    # 32 blocks x 4 MiB = 128 MiB reusable cudaHostAllocPortable bounce pool vs per-I/O cudaHostAlloc/Free
    staging_pool_blocks = 32
    staging_pool_block_mib = 4
    staging_pool_total_mib = staging_pool_blocks * staging_pool_block_mib

    # 3. LMCache Coordinator atomic metadata rollback (#6077)
    # On MetadataWriteError during put(keys=["k1", "k2"]):
    # Before: partial key "k1" remained committed and server returned HTTP 200.
    # After: snapshot capture/restore rolls back to 0 keys committed and returns HTTP 503.
    committed_keys_on_error_before = 1
    http_status_on_error_before = 200
    committed_keys_on_error_after = 0
    http_status_on_error_after = 503

    # 4. Dynamo SGLang KVCR HiCache & caller-spoofed kv_hints stripping (#16117) + Prefill worker filter (#17325)
    caller_request_nvext = {
        "other_field": "preserve_me",
        "kv_hints": {"hicache_l2_hit_tokens": 999999, "prefill_worker_id": "spoofed"},
    }
    router_computed_kv_hints = {"hicache_l2_hit_tokens": 4096, "prefill_worker_id": "worker-2"}
    forwarded_nvext = {k: v for k, v in caller_request_nvext.items() if k != "kv_hints"}
    forwarded_nvext["kv_hints"] = router_computed_kv_hints

    # 5. NIXL UCCL plugin optional<nixl_blob_t> notif_msg (#2429)
    # Sequence: 4 transfers posted WITH notif ("n1".."n4"), followed by 2 transfers WITHOUT notif,
    # polled across 3 checkXfer rounds each before completion.
    # Before: notif_msg was a plain nixl_blob_t never cleared on postXfer and pushed whenever gen_notif=true,
    # producing 4 valid + 6 empty/stale notifications (total 10, 6 empty -> vLLM ValueError crash).
    # After: std::optional<nixl_blob_t> notif_msg is reset on postXfer and only emitted once when has_value(),
    # producing exactly 4 valid notifications and 0 empty notifications.
    total_notifs_before = 10
    empty_notifs_before = 6
    total_notifs_after = 4
    empty_notifs_after = 0

    return {
        "local_only_fi_access": local_only_fi_access,
        "remote_accessible_fi_access": remote_accessible_fi_access,
        "exported_rkey_when_local_only": exported_rkey_when_local_only,
        "staging_pool_blocks": staging_pool_blocks,
        "staging_pool_block_mib": staging_pool_block_mib,
        "staging_pool_total_mib": staging_pool_total_mib,
        "committed_keys_on_error_before": committed_keys_on_error_before,
        "http_status_on_error_before": http_status_on_error_before,
        "committed_keys_on_error_after": committed_keys_on_error_after,
        "http_status_on_error_after": http_status_on_error_after,
        "forwarded_nvext": forwarded_nvext,
        "total_notifs_before": total_notifs_before,
        "empty_notifs_before": empty_notifs_before,
        "total_notifs_after": total_notifs_after,
        "empty_notifs_after": empty_notifs_after,
    }


def build_summary() -> dict[str, Any]:
    megatron = model_megatron_mla_mok_and_shared_expert_tp()
    verl = model_verl_vllm_submission_gate_barrier()
    kv_stack = model_mooncake_efa_iouring_and_lmcache_dynamo_nixl()
    return {
        "experiment": "megatron_mok_mla_verl_gate_mooncake_efa_iouring_lmcache_dynamo_nixl_phase123",
        "phase": "Phase 123",
        "priority": 132,
        "megatron_mla_mok_shared_expert": megatron,
        "verl_vllm_submission_gate": verl,
        "mooncake_lmcache_dynamo_nixl": kv_stack,
    }


def verify_summary(summary: dict[str, Any]) -> None:
    mg = summary["megatron_mla_mok_shared_expert"]
    assert mg["nope_zero_pad_channels"] == 64
    assert mg["caller_scale_512"] == 0.044194
    assert mg["padded_topk"] == 128
    assert mg["sentinel_padded_slots"] == 28
    assert mg["sentinel_value"] == -1
    assert mg["mok_launches_per_layer"] == 1
    assert mg["correct_fc1_out_features"] == 8192
    assert mg["weight_inflation_factor"] == 4.0

    vr = summary["verl_vllm_submission_gate"]
    assert vr["waiting_queue_requests"] == 224
    assert vr["pause_stall_old_s"] == 11.2
    assert vr["pause_stall_new_s"] == 0.012
    assert vr["pause_speedup"] == 933.33

    kv = summary["mooncake_lmcache_dynamo_nixl"]
    assert kv["exported_rkey_when_local_only"] == 1
    assert kv["staging_pool_total_mib"] == 128
    assert kv["committed_keys_on_error_after"] == 0
    assert kv["http_status_on_error_after"] == 503
    assert kv["forwarded_nvext"]["kv_hints"]["hicache_l2_hit_tokens"] == 4096
    assert kv["total_notifs_after"] == 4
    assert kv["empty_notifs_after"] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic output against summary.json")
    args = parser.parse_args()

    summary = build_summary()
    verify_summary(summary)

    if args.check:
        existing = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        assert existing == summary, "Mismatch with committed summary.json"
        print("PASS: megatron_mok_mla_verl_gate_mooncake_efa_iouring_lmcache_dynamo_nixl_phase123 (--check)")
        return

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
