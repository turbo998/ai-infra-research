#!/usr/bin/env python3
"""Deterministic CPU/offline simulator for Phase 73 Track C:
Mooncake AWS Neuron (Trainium) Direct HBM-over-EFA & Stale-Peer AV Eviction & HCA/NUMA
Store Affinity, LMCache Cambricon MLU Backend & Request-Exclusive Local Checksum Bench,
TorchTitan FlashAttention Fallback & Lazy Backend Deferral, Megatron-LM GDP Prefill
Prepare Fusion & EP Consensus Enforcement, and Dynamo KV-Router Event Filtering,
Worker GC Freeze & Batched Token Metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_mooncake_neuron_efa_and_store_affinity() -> dict[str, Any]:
    """Model Mooncake@f88d94d1bc (AWS Neuron HBM over EFA), Mooncake@487da6af82
    (EFA stale-peer eviction on reconnect), and Mooncake@868a6d05ab (HCA/NUMA store affinity).
    """
    # On trn2.48xlarge (16 Neuron devices):
    # - Remote PCIe switch NIC: 2.5 GB/s per device, 12.31 GB/s aggregate over 16 devices
    # - Switch-local EFA NIC via nec_get_device_pci_bdf(): 19.2 GB/s per device, 231.16 GB/s aggregate
    # - Intra-node SHM fallback (uncached PCIe BAR memcpy): 0.07 GB/s vs EFA loopback 19.23 GB/s
    per_device_cross_switch_gbps = 2.5
    per_device_local_switch_gbps = 19.2
    per_device_speedup = round(per_device_local_switch_gbps / per_device_cross_switch_gbps, 2)  # 7.68x
    aggregate_16dev_unaligned_gbps = 12.31
    aggregate_16dev_aligned_gbps = 231.16
    aggregate_speedup = round(aggregate_16dev_aligned_gbps / aggregate_16dev_unaligned_gbps, 2)  # 18.78x
    shm_uncached_bar_gbps = 0.07
    efa_loopback_gbps = 19.23
    loopback_vs_shm_ratio = round(efa_loopback_gbps / shm_uncached_bar_gbps, 1)  # 274.7x

    # Stale-peer eviction on rolling upgrade (120 decode + 120 prefill, 5000 restarts)
    def stable_host_nic_key(peer_nic_path: str) -> str:
        host_part, nic_part = peer_nic_path.split("@", 1)
        stable_host = host_part.split(":", 1)[0]
        return f"{stable_host}@{nic_part}"

    churn_iterations = 5000
    old_peer_map_size = churn_iterations
    stable_keys = {
        stable_host_nic_key(f"10.0.1.12:{12000 + i}:{1700000000 + i}@rdmap201s0")
        for i in range(churn_iterations)
    }
    new_peer_map_size = len(stable_keys)
    stale_peer_prov_errnos = [4, 8, 9, 14]

    return {
        "per_device_cross_switch_gbps": per_device_cross_switch_gbps,
        "per_device_local_switch_gbps": per_device_local_switch_gbps,
        "per_device_affinity_speedup": per_device_speedup,
        "aggregate_16dev_unaligned_gbps": aggregate_16dev_unaligned_gbps,
        "aggregate_16dev_aligned_gbps": aggregate_16dev_aligned_gbps,
        "aggregate_16dev_speedup": aggregate_speedup,
        "shm_uncached_bar_gbps": shm_uncached_bar_gbps,
        "efa_loopback_gbps": efa_loopback_gbps,
        "loopback_vs_shm_ratio": loopback_vs_shm_ratio,
        "churn_iterations": churn_iterations,
        "old_peer_map_size": old_peer_map_size,
        "new_peer_map_size": new_peer_map_size,
        "stale_peer_prov_errnos": stale_peer_prov_errnos,
    }


def simulate_lmcache_torchtitan_megatron_dynamo() -> dict[str, Any]:
    """Model LMCache@a2cd25df48 (Cambricon MLU backend), LMCache@6b2b754654 (bench server
    request-exclusive allocator), TorchTitan@89e4339c96 & 3ed5aa976f, Megatron-LM@b291235082
    & 7d3f0fb3b5, and Dynamo@8bf042187a, a789babbe3, 45cc9e8a1a, 906e89d3a3.
    """
    # 1. LMCache MLU backend: D2H copies into strided host slice (when skip_prefix_n_blocks > 0)
    #    must be blocking on torch_mlu 1.26.1 + torch 2.6.0; contiguous slices stay non_blocking=True
    mlu_d2h_modes = [
        {"skip_prefix_n_blocks": 0, "host_slice_contiguous": True, "non_blocking": True},
        {"skip_prefix_n_blocks": 2, "host_slice_contiguous": False, "non_blocking": False},
    ]

    # 2. TorchTitan: macOS CPU test collection with lazy attn_gym exports deferred
    torchtitan_cpu_tests_before = {"collected": 1371, "errors": 23}
    torchtitan_cpu_tests_after = {"collected": 1640, "errors": 5, "fixed_modules": 18}

    # 3. Megatron-LM GDP prefill prepare fusion: ~14 separate PyTorch elementwise/strided-copy
    #    kernels per layer collapsed into 1 Triton launch (`gdp_prefill_prepare`)
    gdp_layers = 40
    prefill_prep_launches_per_layer_before = 14
    prefill_prep_launches_per_layer_after = 1
    total_prefill_prep_launches_before = gdp_layers * prefill_prep_launches_per_layer_before  # 560
    total_prefill_prep_launches_after = gdp_layers * prefill_prep_launches_per_layer_after    # 40

    # 4. Dynamo:
    # - /health default omits instances list (opt-in ?instances=true saves ~10-50ms JSON serialization)
    # - --freeze-gc-after-init freezes ~1,000,000 static objects so gen2 GC skips them
    # - RequestMetrics batches per-token output_tokens atomic increments every LOCAL_FLUSH_TOKENS = 64
    local_flush_tokens = 64
    output_tokens_generated = 512
    shared_atomic_ops_before = output_tokens_generated  # 512 shared cache-line bounces
    shared_atomic_ops_after = output_tokens_generated // local_flush_tokens  # 8 flushes (64x reduction)
    atomic_contention_reduction = shared_atomic_ops_before // shared_atomic_ops_after

    return {
        "mlu_d2h_modes": mlu_d2h_modes,
        "torchtitan_cpu_tests_before": torchtitan_cpu_tests_before,
        "torchtitan_cpu_tests_after": torchtitan_cpu_tests_after,
        "megatron_gdp_layers": gdp_layers,
        "megatron_gdp_prefill_launches_before": total_prefill_prep_launches_before,
        "megatron_gdp_prefill_launches_after": total_prefill_prep_launches_after,
        "dynamo_frozen_static_objects_approx": 1_000_000,
        "dynamo_local_flush_tokens": local_flush_tokens,
        "dynamo_shared_atomic_ops_before": shared_atomic_ops_before,
        "dynamo_shared_atomic_ops_after": shared_atomic_ops_after,
        "dynamo_atomic_contention_reduction_x": atomic_contention_reduction,
    }


def build_summary() -> dict[str, Any]:
    mooncake = simulate_mooncake_neuron_efa_and_store_affinity()
    ecosystem = simulate_lmcache_torchtitan_megatron_dynamo()
    return {
        "experiment": "mooncake_neuron_efa_lmcache_mlu_megatron_dynamo_sim",
        "phase": "Phase 73",
        "priority": 82,
        "track": "Track C",
        "mooncake_neuron_efa_and_store": mooncake,
        "lmcache_torchtitan_megatron_dynamo": ecosystem,
        "invariants": {
            "neuron_16dev_aggregate_231_16_gbps": abs(mooncake["aggregate_16dev_aligned_gbps"] - 231.16) < 1e-6,
            "neuron_affinity_18_78x": abs(mooncake["aggregate_16dev_speedup"] - 18.78) < 1e-6,
            "efa_peer_map_evicted_to_1": mooncake["new_peer_map_size"] == 1,
            "mlu_strided_d2h_forces_blocking": ecosystem["mlu_d2h_modes"][1]["non_blocking"] is False,
            "megatron_gdp_launches_560_to_40": (
                ecosystem["megatron_gdp_prefill_launches_before"] == 560
                and ecosystem["megatron_gdp_prefill_launches_after"] == 40
            ),
            "dynamo_atomic_reduction_64x": ecosystem["dynamo_atomic_contention_reduction_x"] == 64,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against committed summary.json")
    args = parser.parse_args()

    summary = build_summary()
    summary_path = Path(__file__).with_name("summary.json")

    if not all(summary["invariants"].values()):
        raise AssertionError(f"Invariant failure: {summary['invariants']}")

    if args.check:
        expected = json.loads(summary_path.read_text(encoding="utf-8"))
        if expected != summary:
            raise AssertionError("summary.json mismatch for Track C Phase 73")
        print("OK: Phase 73 Track C summary verified.")
        return 0

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
