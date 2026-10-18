"""Deterministic offline simulator for Operator-Level Selective Activation Rematerialization (SAC) & PCIe Gen5 Offloading.

Models a 70B-class Transformer layer (Hidden D=8192, SwiGLU FFN D_ff=28672, GQA H_q=64, H_kv=8, Layers L=80)
and a DeepSeek-style MoE layer (routed_experts.w13) across sequence lengths:
1. Operator Value-Ratio Knapsack Ranking (`eta_op = Bytes_Saved / Recompute_FLOPs`) for every sub-operator in a Transformer block.
2. No Checkpoint (Save All) vs Full Layer Rematerialization (`+33.3%` step time overhead) vs Megatron Core-Attention-Only SAC vs TorchTitan Operator-Level SAC (`save_regions` / `recompute_regions`) vs SAC + PCIe Gen5 (`64 GB/s` per direction, NUMA-bound pinned host memory `saved_tensors_hooks`) Zero-Bubble Dual-Stream Offloading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.json"


def build_operator_knapsack_table(
    tokens_per_rank: int = 16384,
    hidden_size: int = 8192,
    ffn_hidden_size: int = 28672,
    num_q_heads: int = 64,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    bytes_per_elem: int = 2,
) -> list[dict[str, Any]]:
    """Compute activation memory saved (MiB) vs recomputation FLOPs (TFLOPs) and efficiency ratio eta_op."""
    t = tokens_per_rank
    d = hidden_size
    dff = ffn_hidden_size
    d_kv = num_kv_heads * head_dim

    # Each entry: (op_name, category, saved_bytes, recompute_flops, default_policy)
    raw_ops = [
        (
            "rmsnorm_attn_and_mlp",
            "norm_pointwise",
            2 * t * d * bytes_per_elem,
            2 * 4.0 * t * d,
            "recompute",
        ),
        (
            "rope_q_and_k",
            "positional_pointwise",
            t * (d + d_kv) * bytes_per_elem,
            6.0 * t * (d + d_kv),
            "recompute",
        ),
        (
            "swiglu_gate_act",
            "ffn_pointwise_or_w13_region",
            2 * t * dff * bytes_per_elem,
            6.0 * t * dff,
            "recompute",
        ),
        (
            "residual_dropout_add",
            "residual_pointwise",
            2 * t * d * bytes_per_elem,
            2 * 2.0 * t * d,
            "recompute",
        ),
        (
            "flash_attn_out_and_lse",
            "attention_core",
            t * d * bytes_per_elem + t * num_q_heads * 4,
            2.0 * (t ** 2) * d * 0.5,
            "recompute_or_offload",
        ),
        (
            "qkv_linear_proj",
            "gemm_projection",
            t * (d + 2 * d_kv) * bytes_per_elem,
            2.0 * t * d * (d + 2 * d_kv),
            "save_or_offload",
        ),
        (
            "attn_out_linear_proj",
            "gemm_projection",
            t * d * bytes_per_elem,
            2.0 * t * d * d,
            "save_or_offload",
        ),
        (
            "mlp_up_gate_w13_proj",
            "gemm_ffn_w13",
            2 * t * dff * bytes_per_elem,
            2.0 * t * d * (2 * dff),
            "recompute_in_moe_or_save",
        ),
        (
            "mlp_down_w2_proj",
            "gemm_ffn_w2",
            t * d * bytes_per_elem,
            2.0 * t * dff * d,
            "save_or_offload",
        ),
    ]

    table = []
    for name, cat, s_bytes, r_flops, policy in raw_ops:
        mib = s_bytes / (1024.0 ** 2)
        tflops = r_flops / 1e12
        # eta_op in Bytes saved per GFLOP of recompute
        bytes_per_gflop = s_bytes / (r_flops / 1e9)
        table.append({
            "operator": name,
            "category": cat,
            "saved_mib_per_layer": round(mib, 2),
            "recompute_tflops_per_layer": round(tflops, 4),
            "eta_bytes_per_gflop": round(bytes_per_gflop, 2),
            "recommended_action": policy,
        })

    table.sort(key=lambda x: x["eta_bytes_per_gflop"], reverse=True)
    return table


def simulate_activation_memory_and_step_time(
    num_layers: int = 80,
    tokens_per_rank: int = 16384,
    gpu_tflops_bf16: float = 720.0,
    pcie_gen5_eff_gbps: float = 56.0,
) -> list[dict[str, Any]]:
    """Compare 5 activation management policies across an 80-layer 70B training step."""
    op_table = {row["operator"]: row for row in build_operator_knapsack_table(tokens_per_rank=tokens_per_rank)}

    total_layer_saved_mib = sum(r["saved_mib_per_layer"] for r in op_table.values())
    # Forward GEMM + Attention compute per layer
    fwd_tflops_per_layer = sum(r["recompute_tflops_per_layer"] for r in op_table.values())
    bwd_tflops_per_layer = 2.0 * fwd_tflops_per_layer
    base_step_tflops = num_layers * (fwd_tflops_per_layer + bwd_tflops_per_layer)
    base_compute_ms = (base_step_tflops / gpu_tflops_bf16) * 1e3

    pointwise_ops = [
        "rmsnorm_attn_and_mlp",
        "rope_q_and_k",
        "swiglu_gate_act",
        "residual_dropout_add",
    ]
    knapsack_sac_ops = pointwise_ops + ["flash_attn_out_and_lse"]

    policies = [
        ("no_recompute_save_all", [], 0.0),
        ("full_block_remat_checkpoint", list(op_table.keys()), 0.0),
        ("megatron_core_attn_pointwise_sac", knapsack_sac_ops, 0.0),
        ("torchtitan_sac_recompute_w13_regions", pointwise_ops + ["mlp_up_gate_w13_proj"], 0.0),
        ("sac_knapsack_plus_pcie_gen5_offload", knapsack_sac_ops, 0.72),
    ]

    results = []
    for policy_name, recomputed_list, offload_fraction in policies:
        if policy_name == "full_block_remat_checkpoint":
            # Only layer boundary input hidden_states (t * D * 2 bytes = 256 MiB/layer) kept in HBM
            retained_mib_per_layer = 256.0
            recompute_tflops_per_layer = fwd_tflops_per_layer
        else:
            recomputed_mib = sum(op_table[k]["saved_mib_per_layer"] for k in recomputed_list)
            retained_mib_per_layer = total_layer_saved_mib - recomputed_mib
            recompute_tflops_per_layer = sum(op_table[k]["recompute_tflops_per_layer"] for k in recomputed_list)

        offloaded_mib_per_layer = retained_mib_per_layer * offload_fraction
        hbm_resident_mib_per_layer = retained_mib_per_layer - offloaded_mib_per_layer

        # Add 2 double-buffered staging slots in HBM when offloading is active
        staging_buffer_gib = (2.0 * offloaded_mib_per_layer) / 1024.0 if offload_fraction > 0 else 0.0
        total_hbm_activation_gib = (num_layers * hbm_resident_mib_per_layer) / 1024.0 + staging_buffer_gib
        baseline_hbm_gib = (num_layers * total_layer_saved_mib) / 1024.0
        memory_reduction_pct = (1.0 - total_hbm_activation_gib / baseline_hbm_gib) * 100.0

        recompute_ms = ((num_layers * recompute_tflops_per_layer) / gpu_tflops_bf16) * 1e3

        # PCIe Gen5 D2H during forward + H2D prefetch during backward on independent copy stream
        layer_compute_fwd_ms = (fwd_tflops_per_layer / gpu_tflops_bf16) * 1e3
        layer_compute_bwd_ms = ((bwd_tflops_per_layer + recompute_tflops_per_layer) / gpu_tflops_bf16) * 1e3
        layer_dma_oneway_ms = ((offloaded_mib_per_layer * (1024.0 ** 2)) / (pcie_gen5_eff_gbps * 1e9)) * 1e3

        # Dual-stream overlap hides DMA behind layer compute except boundary stalls
        exposed_d2h_per_layer_ms = max(0.0, layer_dma_oneway_ms - layer_compute_fwd_ms * 0.92)
        exposed_h2d_per_layer_ms = max(0.0, layer_dma_oneway_ms - layer_compute_bwd_ms * 0.94)
        # First/last layer pipeline warmup/drain for async copy stream
        dma_pipeline_tail_ms = 2.0 * layer_dma_oneway_ms * 0.15 if offload_fraction > 0 else 0.0
        exposed_pcie_stall_ms = num_layers * (exposed_d2h_per_layer_ms + exposed_h2d_per_layer_ms) + dma_pipeline_tail_ms

        total_step_ms = base_compute_ms + recompute_ms + exposed_pcie_stall_ms
        step_overhead_pct = ((total_step_ms - base_compute_ms) / base_compute_ms) * 100.0

        results.append({
            "policy": policy_name,
            "retained_mib_per_layer": round(retained_mib_per_layer, 2),
            "offloaded_mib_per_layer": round(offloaded_mib_per_layer, 2),
            "hbm_resident_mib_per_layer": round(hbm_resident_mib_per_layer, 2),
            "total_hbm_activation_gib": round(total_hbm_activation_gib, 2),
            "memory_reduction_pct": round(memory_reduction_pct, 2),
            "base_compute_ms": round(base_compute_ms, 2),
            "recompute_overhead_ms": round(recompute_ms, 2),
            "layer_dma_oneway_ms": round(layer_dma_oneway_ms, 2),
            "exposed_pcie_stall_ms": round(exposed_pcie_stall_ms, 2),
            "total_step_time_ms": round(total_step_ms, 2),
            "step_time_overhead_pct": round(step_overhead_pct, 2),
        })

    return results


def simulate_pcie_bandwidth_sensitivity() -> list[dict[str, Any]]:
    """Compare NUMA-misaligned pageable host memory vs NUMA-bound pinned PCIe Gen4/Gen5 offloading."""
    configs = [
        ("pageable_no_numa_gen4", 18.0),
        ("pinned_cross_numa_gen5", 32.0),
        ("pinned_numa_bound_gen5_x16", 56.0),
        ("nvlink_c2c_grace_hopper_blackwell", 420.0),
    ]
    rows = []
    for label, bw_gbps in configs:
        sim = simulate_activation_memory_and_step_time(pcie_gen5_eff_gbps=bw_gbps)
        sac_off = next(r for r in sim if r["policy"] == "sac_knapsack_plus_pcie_gen5_offload")
        rows.append({
            "host_interconnect_profile": label,
            "effective_dma_bw_gbps": bw_gbps,
            "layer_dma_oneway_ms": sac_off["layer_dma_oneway_ms"],
            "exposed_pcie_stall_ms": sac_off["exposed_pcie_stall_ms"],
            "step_time_overhead_pct": sac_off["step_time_overhead_pct"],
            "total_hbm_activation_gib": sac_off["total_hbm_activation_gib"],
        })
    return rows


def generate_results() -> dict[str, Any]:
    op_table = build_operator_knapsack_table()
    policy_matrix = simulate_activation_memory_and_step_time()
    pcie_sweep = simulate_pcie_bandwidth_sensitivity()

    no_recomp = next(r for r in policy_matrix if r["policy"] == "no_recompute_save_all")
    full_remat = next(r for r in policy_matrix if r["policy"] == "full_block_remat_checkpoint")
    sac_only = next(r for r in policy_matrix if r["policy"] == "megatron_core_attn_pointwise_sac")
    sac_w13 = next(r for r in policy_matrix if r["policy"] == "torchtitan_sac_recompute_w13_regions")
    sac_offload = next(r for r in policy_matrix if r["policy"] == "sac_knapsack_plus_pcie_gen5_offload")

    return {
        "experiment_id": "selective_activation_offload",
        "title": "Operator-Level Selective Activation Rematerialization (SAC) & PCIe Gen5 Zero-Bubble Offloading",
        "model_profile": {
            "architecture": "80-Layer 70B GQA Transformer / MoE-w13 Region Profile",
            "tokens_per_rank": 16384,
            "hidden_size": 8192,
            "ffn_hidden_size": 28672,
            "precision": "BF16 (2 bytes/element)",
        },
        "operator_knapsack_ranking": op_table,
        "policy_comparison": policy_matrix,
        "interconnect_sensitivity": pcie_sweep,
        "headline_findings": {
            "baseline_save_all_hbm_gib": no_recomp["total_hbm_activation_gib"],
            "full_remat_step_overhead_pct": full_remat["step_time_overhead_pct"],
            "sac_only_memory_reduction_pct": sac_only["memory_reduction_pct"],
            "sac_only_step_overhead_pct": sac_only["step_time_overhead_pct"],
            "torchtitan_w13_sac_memory_reduction_pct": sac_w13["memory_reduction_pct"],
            "torchtitan_w13_sac_step_overhead_pct": sac_w13["step_time_overhead_pct"],
            "sac_plus_pcie_gen5_hbm_gib": sac_offload["total_hbm_activation_gib"],
            "sac_plus_pcie_gen5_memory_reduction_pct": sac_offload["memory_reduction_pct"],
            "sac_plus_pcie_gen5_step_overhead_pct": sac_offload["step_time_overhead_pct"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate deterministic results against results.json")
    args = parser.parse_args()

    results = generate_results()
    serialized = json.dumps(results, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not RESULTS_PATH.exists():
            raise SystemExit(f"Missing {RESULTS_PATH}; run without --check first.")
        existing = RESULTS_PATH.read_text(encoding="utf-8")
        if existing != serialized:
            raise SystemExit("Deterministic check failed: generated results differ from results.json")
        print("selective_activation_offload check PASSED")
        return 0

    RESULTS_PATH.write_text(serialized, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
