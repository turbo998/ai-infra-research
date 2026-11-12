from __future__ import annotations

import json
from pathlib import Path


def simulate_ktransformers_amx_numa_selection() -> dict:
    """Deterministic simulator for Hybrid GPU+CPU MoE Inference (KTransformers, Intel AMX & NUMA Binding)."""
    hardware_profile = {
        "model": "DeepSeek-R1 / V3 671B MoE (37B Active/Token, 256 Routed + 1 Shared Expert/Layer, 61 Layers)",
        "gpu": "1x 48GB L40S / RTX 6000 Ada (or 2x 24GB RTX 4090)",
        "cpu": "Dual-Socket Xeon Sapphire/Emerald Rapids (24-Channel DDR5-5600, 760 GB/s Peak BW, Intel AMX)",
        "system_ram_gb": 512,
        "pcie_bw_gb_s": 32.0,
        "provenance_type": "synthetic_combined_with_anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
    }

    scenarios = [
        {
            "id": "S1_Naive_PCIe_Expert_Weight_Swap",
            "name": "Naive On-Demand PCIe Expert Weight Swapping (Accelerate / DeepSpeed Zero-Inference)",
            "placement": "Move 16.5 GB Active Expert Weights CPU->GPU over PCIe per Token",
            "pcie_traffic_per_token_mb": 16500.0,
            "upi_cross_socket_stall_pct": 42.0,
            "prefill_tok_s": 14.5,
            "decode_tok_s": 1.9,
            "decode_speedup": 1.00,
            "server_capex_usd": 22000,
            "decision_action": "Reject (515 ms/tok PCIe Bus Saturation Collapse)",
        },
        {
            "id": "S2_Default_CPU_MoE_Unpinned_NUMA",
            "name": "Standard Hybrid GPU-Attn + CPU-MoE Without NUMA Striping (Default llama.cpp)",
            "placement": "Move 14.3 KB Hidden State; OS Default Page Placement Across 2 Sockets",
            "pcie_traffic_per_token_mb": 1.75,
            "upi_cross_socket_stall_pct": 48.5,
            "prefill_tok_s": 48.0,
            "decode_tok_s": 5.4,
            "decode_speedup": 2.84,
            "server_capex_usd": 22000,
            "decision_action": "Reject (Cross-Socket UPI 65 GB/s Bottleneck & Python Dispatch Overhead)",
        },
        {
            "id": "S3_KTransformers_AVX512_NUMA_Striped",
            "name": "KTransformers NUMA-Striped Sub-Expert TP + AVX-512 + CUDA Graph",
            "placement": "GPU MLA/Shared + 50/50 NUMA-Bound Cold Experts + Single-Launch CUDA Graph",
            "pcie_traffic_per_token_mb": 1.75,
            "upi_cross_socket_stall_pct": 1.8,
            "prefill_tok_s": 164.0,
            "decode_tok_s": 16.8,
            "decode_speedup": 8.84,
            "server_capex_usd": 22000,
            "decision_action": "Act Automatically on AMD EPYC / AVX-512 Servers",
        },
        {
            "id": "S4_KTransformers_AMX_NUMA_HotExpert_Overlap",
            "name": "KTransformers Intel AMX Tile Kernel + NUMA Striping + Top-16 GPU Hot-Expert Cache",
            "placement": "GPU MLA/Shared/Top-16 Hot Experts Overlapped w/ Dual-Socket AMX INT4/BF16 Cold Experts",
            "pcie_traffic_per_token_mb": 1.42,
            "upi_cross_socket_stall_pct": 0.9,
            "prefill_tok_s": 285.0,
            "decode_tok_s": 26.8,
            "decode_speedup": 14.11,
            "server_capex_usd": 22000,
            "decision_action": "Act Automatically (Best-in-Class Private 671B Workstation: -87.8% CapEx)",
        },
        {
            "id": "S5_Pure_GPU_8xH800_Reference",
            "name": "Pure 8xH800 / H100 640GB VRAM Node (All 671B Weights in HBM)",
            "placement": "100% GPU HBM3 (26.8 TB/s Aggregate HBM Bandwidth)",
            "pcie_traffic_per_token_mb": 0.0,
            "upi_cross_socket_stall_pct": 0.0,
            "prefill_tok_s": 2450.0,
            "decode_tok_s": 58.0,
            "decode_speedup": 30.53,
            "server_capex_usd": 180000,
            "decision_action": "Confirm (Use Only for High-Concurrency B>=32 Production Clusters)",
        },
    ]

    perturbation_checks = [
        {
            "case": "Cross-Socket NUMA Memory Placement (OS First-Touch vs 50/50 Sub-Expert Tensor Striping)",
            "metric": "Effective DDR5 Memory Bandwidth Utilized (GB/s)",
            "unoptimized": 268.0,
            "ktransformers_amx": 692.0,
            "delta_improvement": "+424.0 GB/s (2.58x Bandwidth via Zero UPI Cross-Traffic)",
        },
        {
            "case": "Python Per-Layer CPU/GPU Launch Overhead Across 61 MoE Layers (Eager vs CUDA Graph)",
            "metric": "671B Decode Throughput (tok/s)",
            "unoptimized": 14.2,
            "ktransformers_amx": 26.8,
            "delta_improvement": "+12.6 tok/s (1.89x Faster by Eliminating 366 Python Kernel Launches/Token)",
        },
        {
            "case": "Prefill Compute Bottleneck at 4K Prompt (AVX-512 Vector VNNI vs Intel AMX 2D Tile Matrix)",
            "metric": "4K Prompt Prefill Speed (tok/s)",
            "unoptimized": 92.0,
            "ktransformers_amx": 285.0,
            "delta_improvement": "+193.0 tok/s (3.10x Faster Prefill via AMX TMUL 16x64 Tile Registers)",
        },
        {
            "case": "Zipfian Expert Routing Skew on Coding Prompts (Caching Top-16 Hot Experts per Layer in GPU VRAM)",
            "metric": "CPU Expert Active Compute Load Reduction (%)",
            "unoptimized": 0.0,
            "ktransformers_amx": 24.5,
            "delta_improvement": "-24.5% CPU Load (Top 6.25% Hot Experts Absorb 24.5% of Routed Tokens on GPU)",
        },
    ]

    latency_scaling = [
        {
            "batch_size": 1,
            "default_llamacpp_tok_s": 5.4,
            "ktransformers_amx_tok_s": 26.8,
        },
        {
            "batch_size": 2,
            "default_llamacpp_tok_s": 8.1,
            "ktransformers_amx_tok_s": 41.2,
        },
        {
            "batch_size": 4,
            "default_llamacpp_tok_s": 11.2,
            "ktransformers_amx_tok_s": 62.5,
        },
        {
            "batch_size": 8,
            "default_llamacpp_tok_s": 13.5,
            "ktransformers_amx_tok_s": 84.0,
        },
    ]

    # Deterministic payload ratio math verification:
    # Moving active weights (8 experts * 58 MoE layers * ~35.5 MB = 16,500 MB)
    # vs moving activations (58 MoE layers * 2 directions * 7168 dims * 2 bytes = ~1.66-1.75 MB per token)
    weight_swap_mb = 16500.0
    activation_transfer_mb = 1.75
    pcie_payload_reduction_ratio = round(weight_swap_mb / activation_transfer_mb, 1)

    summary = {
        "best_workstation_scenario": "S4_KTransformers_AMX_NUMA_HotExpert_Overlap",
        "decode_speedup_vs_weight_swap": round(26.8 / 1.9, 2),
        "decode_speedup_vs_default_cpu_moe": round(26.8 / 5.4, 2),
        "prefill_speedup_vs_default_cpu_moe": round(285.0 / 48.0, 2),
        "pcie_payload_reduction_ratio": pcie_payload_reduction_ratio,
        "capex_savings_vs_8xh800_pct": round((180000 - 22000) / 180000 * 100.0, 2),
    }

    return {
        "experiment_id": "hybrid_gpu_cpu_moe_inference_ktransformers_amx_numa_selection_20261112",
        "hardware_profile": hardware_profile,
        "scenarios": scenarios,
        "perturbation_checks": perturbation_checks,
        "latency_scaling": latency_scaling,
        "summary": summary,
    }


def main() -> None:
    payload = simulate_ktransformers_amx_numa_selection()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} with best scenario {payload['summary']['best_workstation_scenario']}")


if __name__ == "__main__":
    main()
