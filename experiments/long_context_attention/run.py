"""CPU-only offline model for 1M-token long-context hybrid attention, Context Parallelism, and FlashMLA KV layouts.

This experiment does not execute CUDA or Ascend kernels. It evaluates:
1. Exact per-token KV cache byte layouts across Standard GQA-8 BF16, DeepSeek-V3 MLA BF16 (1152 B),
   DeepSeek-V3.2 Hopper FP8 MLA (656 B), DeepSeek-V4.1 SM100 FP8 MLA (528 B), and V4.1 FP4 MLA (288 B)
   as defined in deepseek-ai/FlashMLA `csrc/cuda_kernels/kv_cache_format.h` and
   `docs/20250929-hopper-fp8-sparse-deep-dive.md`.
2. Context Parallelism (Ring Attention P2P overlap vs DeepSpeed-Ulysses All-to-All head sharding)
   communication volume and overlap efficiency at 1M tokens.
3. Hopper SM90 CUDA-core FP8->BF16 dequantization cycle bottleneck (50 cycles vs 34 Tensor Core MMA
   cycles per KV token) and 2-CTA Distributed Shared Memory (DSM) Crossover speedup (halving per-CTA
   dequantization to 25 cycles).
4. Hybrid Mamba-2 + Attention memory pool sizing (`MambaSlotAllocator` per-request state vs
   `Int8CheckpointStore` per-node Radix prefix checkpointing in SGLang `mamba_checkpoint_pool.py`),
   including replay re-prefill latency when an arbitrary prefix truncation misses checkpoint intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input.json"
DEFAULT_JSON = ROOT / "results.json"
DEFAULT_CSV = ROOT / "results.csv"


def load_input(path: Path = DEFAULT_INPUT) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_kv_formats(spec: dict[str, Any]) -> list[dict[str, Any]]:
    seq_len = int(spec["seq_len_tokens"])
    num_layers = int(spec["num_layers"])
    hbm_bw_gbps = float(spec["decode_effective_hbm_bw_gbps"])
    rows: list[dict[str, Any]] = []

    for fmt in spec["kv_formats"]:
        raw_bytes = int(fmt["raw_data_bytes_per_token"])
        scale_bytes = int(fmt["scale_bytes_per_token"])
        rope_bf16_bytes = int(fmt.get("unquantized_rope_bytes_per_token", 0))
        bytes_per_token_layer = raw_bytes + scale_bytes + rope_bf16_bytes
        if bytes_per_token_layer != int(fmt["expected_bytes_per_token"]):
            raise ValueError(
                f"Format {fmt['id']} byte mismatch: computed {bytes_per_token_layer}, "
                f"expected {fmt['expected_bytes_per_token']}"
            )
        total_bytes_1m = seq_len * num_layers * bytes_per_token_layer
        total_gib_1m = total_bytes_1m / (1024 ** 3)
        topk = fmt.get("sparse_topk")
        attended_tokens = int(topk) if topk is not None else seq_len
        attended_bytes_per_step = attended_tokens * num_layers * bytes_per_token_layer
        decode_kv_hbm_ms = (attended_bytes_per_step / (hbm_bw_gbps * 1e9)) * 1000.0
        compression_vs_gqa = float(spec["gqa_baseline_bytes_per_token"]) / bytes_per_token_layer

        rows.append(
            {
                "id": fmt["id"],
                "label": fmt["label"],
                "bytes_per_token_per_layer": bytes_per_token_layer,
                "raw_data_bytes_per_token": raw_bytes,
                "scale_bytes_per_token": scale_bytes,
                "unquantized_rope_bytes_per_token": rope_bf16_bytes,
                "seq_len_tokens": seq_len,
                "num_layers": num_layers,
                "kv_cache_gib_per_1m_req": round(total_gib_1m, 4),
                "attended_tokens_per_decode_step": attended_tokens,
                "decode_kv_hbm_read_mib": round(attended_bytes_per_step / (1024 ** 2), 4),
                "decode_kv_hbm_latency_ms": round(decode_kv_hbm_ms, 4),
                "compression_vs_gqa_bf16": round(compression_vs_gqa, 4),
            }
        )
    return rows


def evaluate_hopper_dsm_crossover(spec: dict[str, Any]) -> dict[str, Any]:
    q_heads_per_cta = int(spec["q_heads_per_cta"])
    d_qk = int(spec["d_qk"])
    d_v = int(spec["d_v"])
    mma_flops_per_cycle_per_sm = int(spec["mma_flops_per_cycle_per_sm"])
    quant_dims = int(spec["quant_dims"])

    mma_flops_per_kv_token = q_heads_per_cta * (d_qk + d_v) * 2
    mma_cycles_per_kv_token = mma_flops_per_kv_token / mma_flops_per_cycle_per_sm

    # Hopper FP8 -> half (1/64) -> float32 (1/64) -> bfloat16 (1/16) + scale multiply (1/256) per element
    dequant_cycles_per_elem = (1.0 / 64.0) + (1.0 / 64.0) + (1.0 / 16.0) + (1.0 / 256.0)
    single_cta_dequant_cycles = quant_dims * dequant_cycles_per_elem
    crossover_cta_dequant_cycles = single_cta_dequant_cycles / 2.0

    single_cta_bottleneck = max(mma_cycles_per_kv_token, single_cta_dequant_cycles)
    crossover_bottleneck = max(mma_cycles_per_kv_token, crossover_cta_dequant_cycles)
    cycle_speedup_ratio = single_cta_bottleneck / crossover_bottleneck

    return {
        "q_heads_per_cta": q_heads_per_cta,
        "mma_flops_per_kv_token": mma_flops_per_kv_token,
        "mma_cycles_per_kv_token": round(mma_cycles_per_kv_token, 4),
        "single_cta_dequant_cycles_per_kv_token": round(single_cta_dequant_cycles, 4),
        "crossover_2cta_dequant_cycles_per_kv_token": round(crossover_cta_dequant_cycles, 4),
        "single_cta_bound": "dequantization-bound" if single_cta_dequant_cycles > mma_cycles_per_kv_token else "mma-bound",
        "crossover_2cta_bound": "mma-bound" if mma_cycles_per_kv_token >= crossover_cta_dequant_cycles else "dequantization-bound",
        "theoretical_cycle_reduction_ratio": round(cycle_speedup_ratio, 4),
        "author_reported_h800_tflops_no_crossover": float(spec["author_reported_h800_tflops_no_crossover"]),
        "author_reported_h800_tflops_with_crossover": float(spec["author_reported_h800_tflops_with_crossover"]),
    }


def evaluate_context_parallelism(spec: dict[str, Any]) -> list[dict[str, Any]]:
    seq_len = int(spec["seq_len_tokens"])
    cp_degree = int(spec["cp_degree"])
    num_layers = int(spec["num_layers"])
    num_q_heads = int(spec["num_q_heads"])
    num_kv_heads_gqa = int(spec["num_kv_heads_gqa"])
    num_kv_heads_mla = int(spec["num_kv_heads_mla"])
    d_head = int(spec["d_head"])
    bpe = int(spec["bytes_per_element"])
    compute_tflops_per_gpu = float(spec["effective_attn_tflops_per_gpu"])
    interconnect_bw_gbps = float(spec["interconnect_unidir_bw_gbps"])

    tokens_per_rank = seq_len // cp_degree
    # Causal attention total FLOPs across all ranks: 2 * S^2 * H_q * d_head (QK^T + SV with 0.5 causal factor)
    # Wait: full causal QK^T + SV is 4 * (S^2 / 2) * H_q * d_head = 2 * S^2 * H_q * d_head per layer
    attn_flops_per_layer_per_gpu = (2.0 * (seq_len ** 2) * num_q_heads * d_head) / cp_degree
    compute_ms_per_layer = (attn_flops_per_layer_per_gpu / (compute_tflops_per_gpu * 1e12)) * 1000.0

    results: list[dict[str, Any]] = []
    for mode in spec["cp_modes"]:
        mode_id = mode["id"]
        if mode["mechanism"] == "ring_p2p":
            kv_heads = num_kv_heads_gqa if mode["kv_type"] == "gqa" else num_kv_heads_mla
            kv_dim = d_head * 2 if mode["kv_type"] == "gqa" else int(mode["mla_kv_dim"])
            # Each ring step sends local KV shard of size (S / CP) * kv_heads * kv_dim * bpe across (CP - 1) steps
            bytes_per_step = tokens_per_rank * kv_heads * kv_dim * bpe
            comm_bytes_per_layer = (cp_degree - 1) * bytes_per_step
            comm_ms_per_layer = (comm_bytes_per_layer / (interconnect_bw_gbps * 1e9)) * 1000.0
            # Ring overlaps P2P with chunk compute; exposed latency is max(compute, comm) + 1 step startup
            step_comm_ms = comm_ms_per_layer / (cp_degree - 1)
            step_compute_ms = compute_ms_per_layer / cp_degree
            exposed_ms_per_layer = step_compute_ms + (cp_degree - 1) * max(step_compute_ms, step_comm_ms)
            head_constraint_ok = True
        elif mode["mechanism"] == "ulysses_all2all":
            kv_heads = num_kv_heads_gqa if mode["kv_type"] == "gqa" else num_kv_heads_mla
            # Ulysses performs 4 All-to-Alls (Q, K, V before attn, O after attn)
            # If CP > kv_heads, KV must be replicated across (CP / kv_heads) ranks during All-to-All
            kv_replication = max(1, cp_degree // kv_heads)
            q_o_bytes = 2 * tokens_per_rank * num_q_heads * d_head * bpe * ((cp_degree - 1) / cp_degree)
            k_v_bytes = 2 * tokens_per_rank * (kv_heads * kv_replication) * d_head * bpe * ((cp_degree - 1) / cp_degree)
            comm_bytes_per_layer = int(q_o_bytes + k_v_bytes)
            comm_ms_per_layer = (comm_bytes_per_layer / (interconnect_bw_gbps * 1e9)) * 1000.0
            # Ulysses All-to-All sits on the critical path before and after attention
            exposed_ms_per_layer = compute_ms_per_layer + comm_ms_per_layer
            head_constraint_ok = cp_degree <= kv_heads
        else:
            raise ValueError(f"Unsupported CP mechanism: {mode['mechanism']}")

        total_prefill_sec = (exposed_ms_per_layer * num_layers) / 1000.0
        comm_to_compute_ratio = comm_ms_per_layer / compute_ms_per_layer

        results.append(
            {
                "id": mode_id,
                "mechanism": mode["mechanism"],
                "kv_type": mode["kv_type"],
                "cp_degree": cp_degree,
                "comm_mib_per_layer_per_gpu": round(comm_bytes_per_layer / (1024 ** 2), 4),
                "compute_ms_per_layer": round(compute_ms_per_layer, 4),
                "comm_ms_per_layer": round(comm_ms_per_layer, 4),
                "exposed_ms_per_layer": round(exposed_ms_per_layer, 4),
                "total_attn_prefill_sec": round(total_prefill_sec, 4),
                "comm_to_compute_ratio": round(comm_to_compute_ratio, 4),
                "zero_kv_replication_within_heads": head_constraint_ok,
            }
        )
    return results


def evaluate_hybrid_mamba_cache(spec: dict[str, Any]) -> dict[str, Any]:
    mamba_layers = int(spec["mamba_layers"])
    attn_layers = int(spec["attn_layers"])
    num_heads = int(spec["mamba_num_heads"])
    d_v = int(spec["mamba_head_v_dim"])
    d_k = int(spec["mamba_head_k_dim"])
    conv_dim = int(spec["mamba_conv_dim"])
    conv_window_minus_1 = int(spec["mamba_conv_window_minus_1"])
    bf16_bpe = 2
    int8_bpe = 1

    # Active MambaPool slot (BF16 temporal state + BF16 conv state across all Mamba layers)
    ssm_elements_per_layer = num_heads * d_v * d_k
    conv_elements_per_layer = conv_dim * conv_window_minus_1
    active_bf16_bytes_per_slot = mamba_layers * (ssm_elements_per_layer + conv_elements_per_layer) * bf16_bpe

    # SGLang MambaCheckpointPool Int8CheckpointStore:
    # qdata: [L, H, d_v, d_k] int8 (1 B) + scale: [L, H, 1, d_k] bf16 (2 B) + conv window bf16
    int8_qdata_bytes = mamba_layers * ssm_elements_per_layer * int8_bpe
    int8_scale_bytes = mamba_layers * (num_heads * 1 * d_k) * bf16_bpe
    conv_bf16_bytes = mamba_layers * conv_elements_per_layer * bf16_bpe
    checkpoint_int8_bytes_per_slot = int8_qdata_bytes + int8_scale_bytes + conv_bf16_bytes

    pool_budget_gib = float(spec["mamba_checkpoint_pool_budget_gib"])
    pool_budget_bytes = int(pool_budget_gib * (1024 ** 3))
    max_slots_bf16 = pool_budget_bytes // active_bf16_bytes_per_slot
    max_slots_int8 = pool_budget_bytes // checkpoint_int8_bytes_per_slot
    slot_capacity_gain = max_slots_int8 / max_slots_bf16

    # Prefix truncation replay analysis:
    # Unlike Paged KV cache where any token boundary can be indexed, Mamba recurrent state can only
    # be restored at saved checkpoint nodes. If a request branches at offset L_branch between checkpoints
    # spaced by `checkpoint_interval_tokens`, the runtime must replay (L_branch % interval) tokens.
    recompute_tokens_per_sec = float(spec["hybrid_prefill_tokens_per_sec"])
    truncation_offset = int(spec["branch_truncation_token_offset"])
    replay_profiles: list[dict[str, Any]] = []
    for interval in spec["checkpoint_intervals_tokens"]:
        interval_int = int(interval)
        restored_token = (truncation_offset // interval_int) * interval_int
        replay_tokens = truncation_offset - restored_token
        replay_ms = (replay_tokens / recompute_tokens_per_sec) * 1000.0
        checkpoints_per_1m = 1048576 // interval_int
        checkpoint_gib_per_1m_int8 = (checkpoints_per_1m * checkpoint_int8_bytes_per_slot) / (1024 ** 3)
        replay_profiles.append(
            {
                "checkpoint_interval_tokens": interval_int,
                "restored_checkpoint_token": restored_token,
                "replay_recompute_tokens": replay_tokens,
                "replay_ttft_penalty_ms": round(replay_ms, 4),
                "checkpoints_per_1m_seq": checkpoints_per_1m,
                "int8_checkpoint_footprint_gib_per_1m": round(checkpoint_gib_per_1m_int8, 4),
            }
        )

    return {
        "mamba_layers": mamba_layers,
        "attn_layers": attn_layers,
        "active_bf16_mib_per_slot": round(active_bf16_bytes_per_slot / (1024 ** 2), 4),
        "checkpoint_int8_mib_per_slot": round(checkpoint_int8_bytes_per_slot / (1024 ** 2), 4),
        "max_cached_nodes_in_budget_bf16": max_slots_bf16,
        "max_cached_nodes_in_budget_int8": max_slots_int8,
        "int8_slot_capacity_multiplier": round(slot_capacity_gain, 4),
        "replay_profiles": replay_profiles,
    }


def run_experiment(spec: dict[str, Any]) -> dict[str, Any]:
    kv_rows = evaluate_kv_formats(spec["kv_section"])
    dsm_crossover = evaluate_hopper_dsm_crossover(spec["hopper_dsm_section"])
    cp_rows = evaluate_context_parallelism(spec["cp_section"])
    mamba_summary = evaluate_hybrid_mamba_cache(spec["hybrid_mamba_section"])

    gqa_row = next(r for r in kv_rows if r["id"] == "gqa8_bf16_dense")
    v41_fp8_row = next(r for r in kv_rows if r["id"] == "deepseek_v4_1_mla_fp8_dsa")
    v41_fp4_row = next(r for r in kv_rows if r["id"] == "deepseek_v4_1_mla_fp4_dsa")

    return {
        "experiment_id": "long_context_attention",
        "measurement_type": "own-offline",
        "Clear_Boundary": "CPU-only analytical model derived from FlashMLA and SGLang source specifications; not a GPU kernel benchmark.",
        "kv_format_comparison": kv_rows,
        "hopper_dsm_crossover": dsm_crossover,
        "context_parallelism_comparison": cp_rows,
        "hybrid_mamba_cache": mamba_summary,
        "headline_findings": {
            "gqa8_1m_kv_gib": gqa_row["kv_cache_gib_per_1m_req"],
            "v41_fp8_1m_kv_gib": v41_fp8_row["kv_cache_gib_per_1m_req"],
            "v41_fp4_1m_kv_gib": v41_fp4_row["kv_cache_gib_per_1m_req"],
            "v41_fp8_dsa_decode_hbm_ms": v41_fp8_row["decode_kv_hbm_latency_ms"],
            "gqa8_dense_decode_hbm_ms": gqa_row["decode_kv_hbm_latency_ms"],
            "hopper_dsm_cycle_speedup": dsm_crossover["theoretical_cycle_reduction_ratio"],
            "mamba_int8_capacity_multiplier": mamba_summary["int8_slot_capacity_multiplier"],
        },
    }


def write_csv(results: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "id",
        "bytes_per_token_per_layer",
        "kv_cache_gib_per_1m_req",
        "attended_tokens_per_decode_step",
        "decode_kv_hbm_read_mib",
        "decode_kv_hbm_latency_ms",
        "compression_vs_gqa_bf16",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["kv_format_comparison"]:
            writer.writerow({k: row[k] for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    spec = load_input(args.input)
    results = run_experiment(spec)

    if args.check:
        expected = json.loads(args.output_json.read_text(encoding="utf-8"))
        if results != expected:
            raise SystemExit("long_context_attention results.json is out of date; re-run without --check.")
        print("long_context_attention check passed.")
        return 0

    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(results, args.output_csv)
    print(f"Wrote {args.output_json} and {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
