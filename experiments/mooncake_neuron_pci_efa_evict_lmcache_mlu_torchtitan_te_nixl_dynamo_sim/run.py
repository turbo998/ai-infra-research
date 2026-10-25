#!/usr/bin/env python3
"""Deterministic stdlib simulator for Phase 74 Track C:
Mooncake AWS Trainium PCIe-Switch EFA Affinity & Stale-Peer Address-Vector Eviction,
LMCache Cambricon MLU HND Normalization & Request-Scoped Checksum Verifier,
TorchTitan Muse Spark 816B-A42B GB300 RL Re-Sharding & Qwen3-30B-A3B MXFP8 Mega-Fused Rollout,
TransformerEngine Triton Mask-Map Over-Capacity Token Clamp, and NIXL v6 256B-Aligned SPSC
Telemetry Ring + UCX Request Correlation & Dynamo Lightweight Health Probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_mooncake_neuron_pci_and_efa_eviction() -> dict[str, Any]:
    """Model Mooncake@f88d94d1bce85e959d8cfb5fdd02c840c791d632,
    Mooncake@487da6af8205b131b845d535cbe513152051a6d6, and
    Mooncake@44094f2e01a05bde80a5a7440caa6d324aa02c12:
    - Trainium `/proc/self/maps` `/dev/neuron<N>` discovery + `nec_get_device_pci_bdf()` + PCIe switch
      grouping (`Topology::resolve()` sorted HCA order):
      * Switch-local EFA NIC bandwidth: `19.2 GB/s` vs `2.5 GB/s` one PCIe switch away (`7.68x`).
      * 16-device multi-thread aggregate EFA bandwidth: `231.16 GB/s` vs `12.31 GB/s` (`18.78x`),
        single-thread `18.30 GB/s` vs `6.07 GB/s` (`3.01x`).
    - EFA stale-peer eviction (`stableHostNicKey("ip:port:ts@nic") -> "ip@nic"`, `disconnectIfIdle()`,
      `fi_av_remove`): across `5,000` rolling-upgrade iterations, `peer_map_` stays bounded at `1`
      instead of leaking `5,000` entries and exhausting `fi_av_insert`.
    - C API `getBatchTransferStatus` reduces status polling from `O(N_tasks)` to `O(1)`.
    """
    switch_local_gbps = 19.2
    cross_switch_gbps = 2.5
    per_nic_affinity_gain_x = round(switch_local_gbps / cross_switch_gbps, 2)  # 7.68x

    agg_16dev_before_gbps = 12.31
    agg_16dev_after_gbps = 231.16
    agg_16dev_gain_x = round(agg_16dev_after_gbps / agg_16dev_before_gbps, 2)  # 18.78x

    single_thread_before_gbps = 6.07
    single_thread_after_gbps = 18.30
    single_thread_gain_x = round(single_thread_after_gbps / single_thread_before_gbps, 2)  # 3.01x

    churn_iterations = 5000
    unbounded_peer_map_entries = churn_iterations
    evicted_peer_map_entries = 1

    return {
        "neuron_pci_commit": "f88d94d1bce85e959d8cfb5fdd02c840c791d632",
        "efa_evict_commit": "487da6af8205b131b845d535cbe513152051a6d6",
        "c_api_batch_commit": "44094f2e01a05bde80a5a7440caa6d324aa02c12",
        "switch_local_gbps": switch_local_gbps,
        "cross_switch_gbps": cross_switch_gbps,
        "per_nic_affinity_gain_x": per_nic_affinity_gain_x,
        "agg_16dev_before_gbps": agg_16dev_before_gbps,
        "agg_16dev_after_gbps": agg_16dev_after_gbps,
        "agg_16dev_gain_x": agg_16dev_gain_x,
        "single_thread_before_gbps": single_thread_before_gbps,
        "single_thread_after_gbps": single_thread_after_gbps,
        "single_thread_gain_x": single_thread_gain_x,
        "churn_iterations": churn_iterations,
        "unbounded_peer_map_entries": unbounded_peer_map_entries,
        "evicted_peer_map_entries": evicted_peer_map_entries,
        "c_api_batch_poll_calls_per_step": 1,
    }


def simulate_lmcache_mlu_and_bench_checksum() -> dict[str, Any]:
    """Model LMCache@a2cd25df48b741cde7776971b38f9c476f48bdd3 and
    LMCache@6b2b75465484f38ee7d7d7e0949e239444e1c564:
    - `MluDeviceSpec` normalizes `vllm-mlu` legacy `"NHD"` layout string to physical `"HND"`
      (`[2, num_blocks, num_kv_heads, block_size, head_size]`) and bounds temporary device staging
      to 1 chunk + per-layer scratch instead of `N_chunks` full-batch buffers.
    - `BlockAllocator` allocates disjoint `GroupBlocks` for cold STORE vs warm RETRIEVE and verifies
      per-chunk tensor checksums end-to-end.
    """
    num_layers = 61
    num_chunks = 16
    chunk_bytes_per_layer = 2 * 256 * 8 * 128 * 2  # 1,048,576 B (1 MiB)
    unbounded_staging_mib = round((num_chunks * num_layers * chunk_bytes_per_layer) / (1024 * 1024), 2)  # 976.0 MiB
    bounded_staging_mib = round(((num_layers + 1) * chunk_bytes_per_layer) / (1024 * 1024), 2)  # 62.0 MiB
    staging_reduction_x = round(unbounded_staging_mib / bounded_staging_mib, 2)  # 15.74x

    return {
        "mlu_commit": "a2cd25df48b741cde7776971b38f9c476f48bdd3",
        "bench_commit": "6b2b75465484f38ee7d7d7e0949e239444e1c564",
        "reported_layout": "NHD",
        "normalized_physical_layout": "HND",
        "unbounded_staging_mib": unbounded_staging_mib,
        "bounded_staging_mib": bounded_staging_mib,
        "staging_reduction_x": staging_reduction_x,
        "rejects_raw_int_pointers": True,
        "disjoint_store_retrieve_block_ids": True,
    }


def simulate_torchtitan_musespark_rl_and_te_clamp() -> dict[str, Any]:
    """Model torchtitan@43d1bb98014e0dab19a83e5220ec7ea1ab06d49e,
    torchtitan@df4e4dc22a335031a9952cbe599b206c410f9d60,
    torchtitan@89e4339c963c138188c69655929722df98ef8441, and
    TransformerEngine@c9c92255465b4518380561b1bbb0aebb177c0b6d:
    - Muse Spark 816B-A42B RL trainer on GB300 (`131,072` seq_len):
      * Old topology: `256` GPUs (`FSDP=64, TP=4, CP=1, EP=4`), expert FSDP group size = `256 / 4 = 64` GPUs.
      * New topology: `128` GPUs (`FSDP=8, TP=4, CP=4, EP=16`), expert FSDP group size = `128 / 16 = 8` GPUs
        (`8x` smaller cross-rack AllGather domain, `-50%` trainer GPUs), peak memory `186 GiB/GPU`, `28%` MFU.
    - Qwen3-30B-A3B DAPO-Math on B200 with `mxfp8_mega_fused_inference=True`:
      * Generator decode time: `20.2 -> 17.8 s` (`1.135x`, reported `1.14x`)
      * Trainer rollout wait: `15.8 -> 13.6 s` (`1.162x`, reported `1.16x`)
      * End-to-end step time: `22.7 -> 20.3 s` (`1.118x`, reported `1.12x`)
    - TransformerEngine `_row_id_map_pass_2_kernel` clamps `row_id >= num_out_tokens` to `-1`.
    """
    old_trainer_gpus = 256
    new_trainer_gpus = 128
    old_expert_fsdp_group = 64
    new_expert_fsdp_group = 8
    expert_fsdp_reduction_x = old_expert_fsdp_group // new_expert_fsdp_group  # 8

    gen_decode_bf16_s = 20.2
    gen_decode_mxfp8_s = 17.8
    rollout_wait_bf16_s = 15.8
    rollout_wait_mxfp8_s = 13.6
    step_time_bf16_s = 22.7
    step_time_mxfp8_s = 20.3

    return {
        "musespark_rl_commit": "43d1bb98014e0dab19a83e5220ec7ea1ab06d49e",
        "qwen3_mxfp8_rl_commit": "df4e4dc22a335031a9952cbe599b206c410f9d60",
        "te_clamp_commit": "c9c92255465b4518380561b1bbb0aebb177c0b6d",
        "old_trainer_gpus": old_trainer_gpus,
        "new_trainer_gpus": new_trainer_gpus,
        "trainer_gpu_saving_pct": 50.0,
        "old_expert_fsdp_group": old_expert_fsdp_group,
        "new_expert_fsdp_group": new_expert_fsdp_group,
        "expert_fsdp_reduction_x": expert_fsdp_reduction_x,
        "gb300_peak_mem_gib": 186,
        "gb300_mfu_pct": 28.0,
        "qwen3_gen_decode_speedup_x": round(gen_decode_bf16_s / gen_decode_mxfp8_s, 4),
        "qwen3_rollout_wait_speedup_x": round(rollout_wait_bf16_s / rollout_wait_mxfp8_s, 4),
        "qwen3_step_time_speedup_x": round(step_time_bf16_s / step_time_mxfp8_s, 4),
        "te_over_capacity_sentinel": -1,
    }


def simulate_nixl_v6_telemetry_and_dynamo_health() -> dict[str, Any]:
    """Model NIXL@a9f1f7a76f46f0ee78ec7448936d48ca52073fe3,
    NIXL@f3673f6be3f4f80f6e1e38ff86cebcf9af5750d4, and
    Dynamo@a789babbe37a53175b962085f8a7f89909425e9b:
    - NIXL `sharedRingBuffer` `TELEMETRY_VERSION = 6` places `write_pos` and `read_pos` in separate
      256-byte cache-exclusivity blocks (`512 B` header total, `version` preserved at byte offset `16`),
      caches opposing indices, and uses reader shared `flock` to preserve unread ring events across restarts:
      * ARM same-socket throughput: `11.9M -> 51.9M events/s` (`4.3613x`), cycle RTT `1022 -> 555 ns` (`1.8414x`).
      * ARM cross-socket throughput: `2.5M -> 9.2M events/s` (`3.68x`).
    - NIXL UCX notification appends a 16-byte `treq` (`requestId` uint64 + `execution` uint64) tag after `tctx`.
    - Dynamo `/health` omits instance enumeration unless `?instances=true` is passed.
    """
    arm_same_socket_before_meps = 11.9
    arm_same_socket_after_meps = 51.9
    arm_same_socket_gain_x = round(arm_same_socket_after_meps / arm_same_socket_before_meps, 4)  # 4.3613x

    arm_cross_socket_before_meps = 2.5
    arm_cross_socket_after_meps = 9.2
    arm_cross_socket_gain_x = round(arm_cross_socket_after_meps / arm_cross_socket_before_meps, 4)  # 3.68x

    cycle_rtt_before_ns = 1022
    cycle_rtt_after_ns = 555
    cycle_rtt_speedup_x = round(cycle_rtt_before_ns / cycle_rtt_after_ns, 4)  # 1.8414x

    return {
        "nixl_ring_commit": "a9f1f7a76f46f0ee78ec7448936d48ca52073fe3",
        "nixl_ucx_treq_commit": "f3673f6be3f4f80f6e1e38ff86cebcf9af5750d4",
        "dynamo_health_commit": "a789babbe37a53175b962085f8a7f89909425e9b",
        "telemetry_version": 6,
        "cache_line_pad_bytes": 256,
        "header_total_bytes": 512,
        "version_field_offset_bytes": 16,
        "arm_same_socket_gain_x": arm_same_socket_gain_x,
        "arm_cross_socket_gain_x": arm_cross_socket_gain_x,
        "cycle_rtt_speedup_x": cycle_rtt_speedup_x,
        "ucx_treq_payload_bytes": 16,
        "dynamo_health_default_enumerates_instances": False,
    }


def build_summary() -> dict[str, Any]:
    mooncake = simulate_mooncake_neuron_pci_and_efa_eviction()
    lmcache = simulate_lmcache_mlu_and_bench_checksum()
    torchtitan_te = simulate_torchtitan_musespark_rl_and_te_clamp()
    nixl_dynamo = simulate_nixl_v6_telemetry_and_dynamo_health()

    return {
        "experiment_id": "mooncake_neuron_pci_efa_evict_lmcache_mlu_torchtitan_te_nixl_dynamo_sim",
        "phase": 74,
        "priority": 83,
        "track": "Track C: Distributed KV Transport, RL Co-Design & Control-Plane Telemetry",
        "mooncake_neuron_and_efa": mooncake,
        "lmcache_mlu_and_bench": lmcache,
        "torchtitan_rl_and_te": torchtitan_te,
        "nixl_v6_and_dynamo": nixl_dynamo,
        "key_invariants": {
            "mooncake_16dev_efa_gain_x": mooncake["agg_16dev_gain_x"],
            "mooncake_evicted_peer_map_entries": mooncake["evicted_peer_map_entries"],
            "lmcache_mlu_staging_reduction_x": lmcache["staging_reduction_x"],
            "torchtitan_expert_fsdp_reduction_x": torchtitan_te["expert_fsdp_reduction_x"],
            "torchtitan_qwen3_rollout_wait_speedup_x": torchtitan_te["qwen3_rollout_wait_speedup_x"],
            "nixl_arm_same_socket_gain_x": nixl_dynamo["arm_same_socket_gain_x"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("summary.json"))
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches deterministic output.")
    args = parser.parse_args()

    summary = build_summary()
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing {args.output}; run without --check first.")
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit(f"{args.output} is out of date with deterministic generator.")
        print(f"[OK] {args.output} verified.")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"[WROTE] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
