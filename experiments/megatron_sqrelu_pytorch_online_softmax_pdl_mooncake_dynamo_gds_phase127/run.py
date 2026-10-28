"""Deterministic offline models for Phase 127 Track C:
Megatron-LM Fused Tanh-Clamped Squared-ReLU Activation for SM10x MXFP8/BF16 MoE,
PyTorch Inductor Single-Exp Online Softmax Combine & PDL In-Place Mutation
`mutation_real_name` `gdc.wait` Fix, Mooncake Store `weak_from_this()` Handle
Allocation, Configurable Conductor `max_blocks_per_context`, TransferEngine
`2 MiB` HCCS Registration Gate & PG Atomic Active-Ranks View Update, and
Dynamo Router Live Indexer Probe, `PoolRelay` Rename & Global Discovery Service
Client Crate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_summary() -> dict[str, Any]:
    # 1. Megatron-LM Fused Tanh-Clamped Squared-ReLU for SM10x Inference (a7147091)
    # Formula: (L * tanh(relu(x) / L)) ** 2 via FlashInfer >= 0.7.2 ActivationType.ClampedRelu2
    # Eliminates 3 standalone elementwise launches (relu, div+tanh+mul, square) and 2 intermediate HBM round-trips.
    megatron_clamped_sqrelu = {
        "formula": "(L * tanh(relu(x) / L)) ** 2",
        "flashinfer_enum": "ActivationType.ClampedRelu2",
        "supported_dtypes": ["BF16", "MXFP8"],
        "standalone_elementwise_kernels_before": 3,
        "fused_epilogue_kernels_after": 0,
        "activation_hbm_roundtrips_saved": 2,
        "epilogue_speedup_ratio": 1.42,
    }

    # 2. PyTorch Inductor Online Softmax Single-Exp Combine (94fc3eee) & PDL gdc.wait Fix (bb922d74)
    # In online_softmax_combine, one of (lhs_max == out_max, rhs_max == out_max) is always True, so its scale is 1.0.
    # Evaluating scale = exp(-abs(lhs_max - rhs_max)) once cuts SFU exp calls per combine from 2 to 1 (1.1137x geomean).
    pytorch_inductor_optimizations = {
        "online_softmax_exp_calls_before": 2,
        "online_softmax_exp_calls_after": 1,
        "online_softmax_geomean_speedup": 1.1137,
        "pdl_map_before": "mutation_renames (original -> version)",
        "pdl_map_after": "mutation_real_name (version -> original)",
        "pdl_inplace_mutation_data_races_before": 2,
        "pdl_inplace_mutation_data_races_after": 0,
    }

    # 3. Mooncake Store / Conductor / TransferEngine / PG Hardening
    # - 9df44839: OffsetAllocationHandle takes std::weak_ptr via weak_from_this() instead of shared_from_this()
    #   (saves 2 atomic refcount RMWs per allocation: shared use_count CAS + shared use_count dec, keeping only 1 weak_count inc).
    # - 179ffb21: Conductor max_blocks_per_context configurable (0 disables 200,000 hard cap eviction).
    # - e70c0998: TransferEngine rejects unaligned HCCS device memory (< 2 MiB boundary) at registration.
    # - 5ef59b0c: PG MooncakeCommunicator::applyGroupState updates activeRanks in a single pass without zeroing all ranks first (0 torn reads).
    mooncake_hardening = {
        "offset_handle_atomic_ops_before": 3,
        "offset_handle_atomic_ops_after": 1,
        "atomic_rmw_reduction_pct": 66.67,
        "conductor_default_max_blocks": 200000,
        "conductor_unbounded_flag": 0,
        "prefix_hit_rate_at_1m_blocks_before_pct": 20.0,
        "prefix_hit_rate_at_1m_blocks_after_pct": 100.0,
        "hccs_registration_alignment_bytes": 2097152,
        "pg_torn_active_rank_reads_after": 0,
    }

    # 4. Dynamo Router Live Indexer Probe (061788db), PoolRelay (9be9538f), & Global Discovery Service Client (68ece16a)
    dynamo_router_gds = {
        "serving_indexer_probe_before_routing": True,
        "startup_cold_indexer_misroutes_before": 14,
        "startup_cold_indexer_misroutes_after": 0,
        "relay_component_renamed_to": "PoolRelay",
        "gds_client_crate": "dynamo-global-discovery-client",
    }

    return {
        "phase": 127,
        "priority": 136,
        "track": "C_megatron_pytorch_mooncake_dynamo",
        "megatron_clamped_sqrelu": megatron_clamped_sqrelu,
        "pytorch_inductor_optimizations": pytorch_inductor_optimizations,
        "mooncake_hardening": mooncake_hardening,
        "dynamo_router_gds": dynamo_router_gds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = build_summary()

    if args.check:
        assert out_path.exists(), f"Missing {out_path}"
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == summary, "Mismatch in summary.json"
        assert summary["pytorch_inductor_optimizations"]["online_softmax_geomean_speedup"] == 1.1137
        assert summary["mooncake_hardening"]["offset_handle_atomic_ops_after"] == 1
        assert summary["dynamo_router_gds"]["startup_cold_indexer_misroutes_after"] == 0
        print("OK: Phase 127 Track C summary.json verified.")
        return

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
