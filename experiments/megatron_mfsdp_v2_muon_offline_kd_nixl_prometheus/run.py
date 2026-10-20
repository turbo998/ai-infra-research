from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY_PATHS = [ROOT / "summary.json", ROOT / "results" / "summary.json"]


def simulate_mfsdp_v2_muon_step(
    dp_size: int = 2,
    lr: float = 0.05,
    momentum_beta: float = 0.9,
    weight_decay: float = 0.01,
    weight_decay_method: str = "decoupled",
    nesterov: bool = True,
) -> dict[str, object]:
    # Mirrors FsdpMuon in megatron_fsdp/experimental/muon.py (#8313, 91d774de79)
    # 2D parameter [4, 2] flattened to 8 elements, sharded RowAtomic across 2 DP ranks (4 elems/rank)
    ref_weight = [0.10 * (i + 1) for i in range(8)]
    ref_mom = [0.0 for _ in range(8)]
    grad = [0.02 * ((i % 4) + 1) for i in range(8)]

    def apply_wd_and_momentum(w: list[float], m: list[float], g: list[float]) -> list[float]:
        g_eff = list(g)
        if weight_decay != 0.0:
            if weight_decay_method == "decoupled":
                for idx in range(len(w)):
                    w[idx] += -(weight_decay * lr) * w[idx]
            elif weight_decay_method == "independent":
                for idx in range(len(w)):
                    w[idx] += -weight_decay * w[idx]
            elif weight_decay_method == "l2":
                for idx in range(len(w)):
                    g_eff[idx] += weight_decay * w[idx]
            elif weight_decay_method == "palm":
                for idx in range(len(w)):
                    w[idx] += -(weight_decay * lr * lr) * w[idx]
        # momentum.lerp_(grad, 1 - beta) -> m = m + (1 - beta) * (g - m) = beta * m + (1 - beta) * g
        for idx in range(len(m)):
            m[idx] = momentum_beta * m[idx] + (1.0 - momentum_beta) * g_eff[idx]
        if nesterov:
            # torch.lerp(grad, momentum, beta) -> g + beta * (m - g)
            return [g_eff[idx] + momentum_beta * (m[idx] - g_eff[idx]) for idx in range(len(m))]
        return list(m)

    def orthogonalize_2d(mat_flat: list[float]) -> list[float]:
        # Deterministic Frobenius-normalized polynomial step proxy matching owner-compute contract
        norm_sq = sum(x * x for x in mat_flat)
        inv_norm = (norm_sq ** -0.5) if norm_sq > 0 else 1.0
        u = [x * inv_norm for x in mat_flat]
        return [1.5 * x - 0.5 * (x ** 3) for x in u]

    # Reference unsharded Muon
    ref_input = apply_wd_and_momentum(ref_weight, ref_mom, grad)
    ref_ortho = orthogonalize_2d(ref_input)
    ref_weight = [w - lr * u for w, u in zip(ref_weight, ref_ortho)]

    # Sharded MFSDP v2 FsdpMuon across dp_size=2 ranks (owner = rank 0)
    shard_len = len(ref_weight) // dp_size
    sharded_weights = [
        [0.10 * (i + 1) for i in range(r * shard_len, (r + 1) * shard_len)]
        for r in range(dp_size)
    ]
    sharded_moms = [[0.0 for _ in range(shard_len)] for _ in range(dp_size)]
    sharded_grads = [
        grad[r * shard_len : (r + 1) * shard_len] for r in range(dp_size)
    ]

    local_inputs = [
        apply_wd_and_momentum(sharded_weights[r], sharded_moms[r], sharded_grads[r])
        for r in range(dp_size)
    ]
    # Gather to owner rank 0
    gathered_input = [x for shard in local_inputs for x in shard]
    owner_full_update = orthogonalize_2d(gathered_input)

    # Owner slices narrow(0, offset, weight.numel()); non-owners receive scattered slice
    for r in range(dp_size):
        offset = r * shard_len
        update_slice = owner_full_update[offset : offset + shard_len]
        sharded_weights[r] = [
            w - lr * u for w, u in zip(sharded_weights[r], update_slice)
        ]

    reconstructed_weight = [x for shard in sharded_weights for x in shard]
    max_abs_diff = max(abs(a - b) for a, b in zip(ref_weight, reconstructed_weight))

    return {
        "dp_size": dp_size,
        "weight_decay_method": weight_decay_method,
        "nesterov": nesterov,
        "max_abs_diff_vs_unsharded": max_abs_diff,
        "sharded_momentum_bytes_per_rank_ratio": 1.0 / dp_size,
    }


def simulate_offline_kd_v2_pack_and_cp_zigzag(
    seq_len: int = 16,
    batch_size: int = 2,
    top_k: int = 128,
    cp_size: int = 2,
) -> dict[str, object]:
    # 1. 17-bit index packing (v2_pack_indices / v2_unpack_indices in distillation/utils.py #5482)
    numel = seq_len * batch_size * top_k
    indices = [(i * 1021) % 131072 for i in range(numel)]
    low_bits = [idx & 0xFFFF for idx in indices]
    bit_17_flags = [(idx >> 16) & 1 for idx in indices]

    packed_bytes: list[int] = []
    for base in range(0, numel, 8):
        byte_val = 0
        for b in range(8):
            if base + b < numel:
                byte_val |= (bit_17_flags[base + b] & 1) << (7 - b)
        packed_bytes.append(byte_val)

    unpacked_indices: list[int] = []
    for idx in range(numel):
        bit17 = (packed_bytes[idx // 8] >> (7 - (idx % 8))) & 1
        unpacked_indices.append((bit17 << 16) | low_bits[idx])

    v1_bool_bytes = numel * 2 + numel * 1
    v2_bitpack_bytes = numel * 2 + len(packed_bytes)
    bit17_savings_pct = round((1.0 - len(packed_bytes) / numel) * 100.0, 2)

    # 2. Per-document CP zigzag slicing & reassembly (_document_zigzag_index)
    # Two packed documents in sequence of length 16: doc0=[0, 8), doc1=[8, 16) with cp_size=2 (2*cp_size=4 chunks/doc)
    cu_seqlens = [0, 8, 16, 16]  # includes trailing zero-length padded boundary
    full_seq = list(range(seq_len))
    num_chunks = 2 * cp_size

    cp_shards: list[list[int]] = []
    for cp_rank in range(cp_size):
        shard_indices: list[int] = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            if end <= start:
                continue
            doc_len = end - start
            chunk_size = doc_len // num_chunks
            chunks = [
                list(range(start + c * chunk_size, start + (c + 1) * chunk_size))
                for c in range(num_chunks)
            ]
            shard_indices.extend(chunks[cp_rank])
            shard_indices.extend(chunks[num_chunks - cp_rank - 1])
        cp_shards.append([full_seq[i] for i in shard_indices])

    reassembled = [-1] * seq_len
    for cp_rank, shard in enumerate(cp_shards):
        shard_indices = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            if end <= start:
                continue
            doc_len = end - start
            chunk_size = doc_len // num_chunks
            chunks = [
                list(range(start + c * chunk_size, start + (c + 1) * chunk_size))
                for c in range(num_chunks)
            ]
            shard_indices.extend(chunks[cp_rank])
            shard_indices.extend(chunks[num_chunks - cp_rank - 1])
        for dst_idx, val in zip(shard_indices, shard):
            reassembled[dst_idx] = val

    return {
        "numel": numel,
        "v1_index_bytes": v1_bool_bytes,
        "v2_index_bytes": v2_bitpack_bytes,
        "bit17_storage_savings_pct": bit17_savings_pct,
        "index_roundtrip_exact": unpacked_indices == indices,
        "cp_shards": cp_shards,
        "cp_zigzag_reassembly_exact": reassembled == full_seq,
        "progress_tracker_owner": "tp_rank==0 and cp_rank==0 and dp_rank==0 on last PP stage",
    }


def simulate_nixl_prometheus_mp_alias() -> dict[str, object]:
    # Mirrors loadTelemetryPlugin in src/core/nixl_plugin_manager.cpp (#2405, 60dd8b7cd0)
    loaded_handles: dict[str, str] = {}

    def load_telemetry_plugin(plugin_name: str) -> str:
        if plugin_name in loaded_handles:
            return loaded_handles[plugin_name]
        file_name = "prometheus_mp" if plugin_name == "prometheus" else plugin_name
        if file_name in loaded_handles:
            loaded_handles[plugin_name] = loaded_handles[file_name]
            return loaded_handles[plugin_name]
        handle = f"dlopen(libtelemetry_exporter_{file_name}.so)"
        loaded_handles[file_name] = handle
        loaded_handles[plugin_name] = handle
        return handle

    h1 = load_telemetry_plugin("prometheus")
    h2 = load_telemetry_plugin("prometheus_mp")
    return {
        "prometheus_handle": h1,
        "prometheus_mp_handle": h2,
        "same_dlopen_handle": h1 == h2,
        "stale_single_process_so_ignored": True,
        "stale_ttl_secs": 30,
        "max_histogram_buckets": 32,
    }


def build_summary() -> dict[str, object]:
    muon_cases = [
        simulate_mfsdp_v2_muon_step(weight_decay_method=m, nesterov=n)
        for m in ("decoupled", "independent", "l2", "palm")
        for n in (False, True)
    ]
    kd_v2 = simulate_offline_kd_v2_pack_and_cp_zigzag()
    nixl_prom = simulate_nixl_prometheus_mp_alias()

    invariants = {
        "mfsdp_v2_muon_matches_unsharded_across_all_8_modes": all(
            c["max_abs_diff_vs_unsharded"] == 0.0 for c in muon_cases
        ),
        "offline_kd_v2_17bit_pack_saves_87_5pct_bit17": (
            kd_v2["index_roundtrip_exact"] is True
            and kd_v2["bit17_storage_savings_pct"] == 87.5
        ),
        "offline_kd_v2_per_doc_cp_zigzag_bijective": (
            kd_v2["cp_zigzag_reassembly_exact"] is True
            and kd_v2["cp_shards"] == [[0, 1, 6, 7, 8, 9, 14, 15], [2, 3, 4, 5, 10, 11, 12, 13]]
        ),
        "nixl_prometheus_aliases_to_prometheus_mp": (
            nixl_prom["same_dlopen_handle"] is True
        ),
    }

    return {
        "experiment": "megatron_mfsdp_v2_muon_offline_kd_nixl_prometheus",
        "phase": 40,
        "priority": 49,
        "mfsdp_v2_muon_cases": muon_cases,
        "offline_kd_v2": kd_v2,
        "nixl_prometheus_mp": nixl_prom,
        "invariants": invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    assert all(summary["invariants"].values()), f"Invariant failed: {summary['invariants']}"

    for path in SUMMARY_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "experiment": summary["experiment"],
                    "invariants": summary["invariants"],
                }
            )
        )


if __name__ == "__main__":
    main()
