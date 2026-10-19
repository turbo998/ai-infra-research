#!/usr/bin/env python3
"""Deterministic simulator for Wafer-Scale/SRAM vs GPU HBM Cascaded Speculative Serving.

Models:
1. Pure On-Chip SRAM Engines (Cerebras WSE-3 44 GB SRAM @ 21 PB/s, Groq LPU Rack 14.7 GB SRAM @ 80 TB/s/chip)
   vs GPU HBM3e/HBM4 Supernodes across context lengths L_kv in {4K, 16K, 64K, 128K}:
   quantifying the SRAM KV Capacity Wall when concurrent long-context sessions grow.
2. vLLM V1 Speculative Proposers (vllm/v1/spec_decode/suffix_decoding.py SuffixDecodingProposer
   and vllm/v1/spec_decode/dflash.py DFlashProposer) vs Heterogeneous SRAM-Draft + HBM-Target
   Cascaded Verification (Together / Fireworks style disaggregated speculative cascade).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RESULTS_PATH = Path(__file__).with_name("results.json")


def simulate_sram_vs_hbm_kv_wall() -> list[dict[str, Any]]:
    """Compare max concurrent sessions & TPOT across context lengths (Llama-3.3-70B FP8)."""
    # Llama-3.3-70B FP8 GQA-8 KV Cache: 80 layers * 2 * 8 heads * 128 dim * 1B = 163,840 B/token
    kv_bytes_per_token = 163840

    # 1. Groq LPU 64-Chip Pod: 64 * 230 MB = 14.72 GB total SRAM -> requires 5-pod (320 LPUs = 73.6 GB SRAM)
    #    Leaving only ~3.6 GiB SRAM headroom for activations + KV Cache without re-streaming weights!
    groq_320_free_kv_gib = 3.60

    # 2. Cerebras CS-3 (WSE-3): 44 GB on-wafer SRAM (with weight streaming / 2x CS-3 cluster = 88 GB SRAM,
    #    leaving ~18.0 GiB on-wafer SRAM for ultra-low-latency KV before external MemoryX spill)
    wse3_2x_free_kv_gib = 18.00

    # 3. Blackwell GB200 NVL72 (8-GPU slice for 70B): 8 * 192 GB = 1,536 GiB HBM3e -> 1,440 GiB KV headroom
    gb200_8gpu_free_kv_gib = 1440.0

    context_lengths = [4096, 16384, 65536, 131072]
    rows: list[dict[str, Any]] = []

    for seq_len in context_lengths:
        session_kv_gib = (seq_len * kv_bytes_per_token) / (1024**3)
        groq_max_sessions = max(1, int(groq_320_free_kv_gib // session_kv_gib))
        wse3_onwafer_sessions = max(1, int(wse3_2x_free_kv_gib // session_kv_gib))
        gb200_max_sessions = int(gb200_8gpu_free_kv_gib // session_kv_gib)

        # TPOT at 16 concurrent sessions: if 16 > on-chip SRAM capacity, external MemoryX / host spill
        # degrades bandwidth from 21 PB/s (or 80 TB/s) down to off-wafer/PCIe bandwidth!
        groq_tpot_16sess_ms = (
            round(1.15 + 0.08 * (seq_len / 4096), 2)
            if 16 <= groq_max_sessions
            else round(1.15 + (16 - groq_max_sessions) * (seq_len / 4096) * 2.85, 2)
        )
        wse3_tpot_16sess_ms = (
            round(0.58 + 0.05 * (seq_len / 4096), 2)
            if 16 <= wse3_onwafer_sessions
            else round(0.58 + (16 - wse3_onwafer_sessions) * (seq_len / 4096) * 1.45, 2)
        )
        gb200_standard_tpot_16sess_ms = round(3.85 + 0.22 * (seq_len / 4096), 2)

        rows.append(
            {
                "context_len": seq_len,
                "session_kv_gib": round(session_kv_gib, 4),
                "groq_320lpu_max_sram_sessions": groq_max_sessions,
                "wse3_2x_max_sram_sessions": wse3_onwafer_sessions,
                "gb200_8gpu_max_hbm_sessions": gb200_max_sessions,
                "groq_tpot_at_16_sessions_ms": groq_tpot_16sess_ms,
                "wse3_tpot_at_16_sessions_ms": wse3_tpot_16sess_ms,
                "gb200_ar_tpot_at_16_sessions_ms": gb200_standard_tpot_16sess_ms,
            }
        )
    return rows


def simulate_speculative_cascade_architectures() -> list[dict[str, Any]]:
    """Compare 5 serving architectures on 64K-context Agentic Coding (BS=32, 70B Target)."""
    # Workload: 32 concurrent 64K-context agentic coding / tool-use sessions
    # KV Footprint: 32 * 65536 * 160 KiB = 320 GiB KV Cache
    architectures = [
        {
            "arch_id": "gpu_hbm_autoregressive",
            "label": "GPU HBM3e Autoregressive Baseline (8x H100/B200 TP=8, No Speculation)",
            "draft_step_ms": 0.0,
            "verify_forward_ms": 11.40,
            "expected_accepted_tokens": 1.00,
            "extra_draft_vram_gib": 0.0,
            "hourly_pod_cost_usd": 32.0,
        },
        {
            "arch_id": "vllm_suffix_decoding_cpu",
            "label": "vLLM V1 SuffixDecodingProposer (Zero-Weight CPU Suffix Tree, ArcticInference)",
            "draft_step_ms": 0.18,  # CPU pattern match on recent max_tree_depth tokens
            "verify_forward_ms": 12.60,
            "expected_accepted_tokens": 2.85,  # High match on repetitive code/JSON edits
            "extra_draft_vram_gib": 0.0,  # Zero GPU weight overhead (load_model is no-op)
            "hourly_pod_cost_usd": 32.0,
        },
        {
            "arch_id": "vllm_dflash_block_proposer",
            "label": "vLLM V1 DFlashProposer (Parallel Cross-Attention Query Block Drafting, K=8)",
            "draft_step_ms": 1.42,  # Single parallel non-causal forward pass for 1+K query tokens
            "verify_forward_ms": 13.10,
            "expected_accepted_tokens": 4.15,
            "extra_draft_vram_gib": 4.8,
            "hourly_pod_cost_usd": 32.0,
        },
        {
            "arch_id": "pure_sram_wafer_offload_spill",
            "label": "Pure Wafer/LPU SRAM Cluster + Off-Wafer MemoryX KV Spill (320 GiB KV @ 64K)",
            "draft_step_ms": 0.0,
            "verify_forward_ms": 19.80,  # Stalled by off-wafer KV streaming across 32x 64K sessions
            "expected_accepted_tokens": 1.00,
            "extra_draft_vram_gib": 0.0,
            "hourly_pod_cost_usd": 68.0,
        },
        {
            "arch_id": "sram_draft_hbm_target_cascade",
            "label": "Heterogeneous Cascade: Wafer/SRAM 8B Draft (Sliding-8K) + GPU HBM3e 70B Verifier (Full-64K)",
            "draft_step_ms": 0.64,  # 8 draft tokens @ 0.08 ms/tok in on-chip SRAM (8K sliding KV = 8 GiB fits in SRAM!)
            "verify_forward_ms": 13.20,  # GPU HBM3e verifies all 8 tokens against full 64K KV in 1 pass
            "expected_accepted_tokens": 5.40,
            "extra_draft_vram_gib": 0.0,  # Draft lives on dedicated SRAM sidecar
            "hourly_pod_cost_usd": 38.5,
        },
    ]

    batch_size = 32
    rows: list[dict[str, Any]] = []
    base_tpot = 0.0
    for idx, a in enumerate(architectures):
        step_wall_ms = round(a["draft_step_ms"] + a["verify_forward_ms"], 3)
        effective_tpot_ms = round(step_wall_ms / a["expected_accepted_tokens"], 3)
        if idx == 0:
            base_tpot = effective_tpot_ms
        per_user_tok_s = round(1000.0 / effective_tpot_ms, 2)
        pod_tok_s = round(batch_size * per_user_tok_s, 1)
        cost_per_1m_usd = round(
            (a["hourly_pod_cost_usd"] / 3600.0) * (1e6 / pod_tok_s), 4
        )
        rows.append(
            {
                **a,
                "step_wall_ms": step_wall_ms,
                "effective_tpot_ms": effective_tpot_ms,
                "per_user_tok_per_s": per_user_tok_s,
                "pod_total_tok_per_s": pod_tok_s,
                "cost_per_1m_tokens_usd": cost_per_1m_usd,
                "tpot_speedup_vs_ar_x": round(base_tpot / effective_tpot_ms, 2),
            }
        )
    return rows


def build_results() -> dict[str, Any]:
    kv_wall = simulate_sram_vs_hbm_kv_wall()
    cascades = simulate_speculative_cascade_architectures()

    c64k = next(r for r in kv_wall if r["context_len"] == 65536)
    a_ar = cascades[0]
    a_suffix = cascades[1]
    a_dflash = cascades[2]
    a_cascade = cascades[4]

    return {
        "experiment_id": "wafer_sram_gpu_cascaded_serving",
        "status": "own-offline",
        "source_contracts": [
            "vllm-project/vllm@973dbf8071f8bdd6dd7eb2d6ee0da7cbe6a93670:vllm/v1/spec_decode/suffix_decoding.py",
            "vllm-project/vllm@973dbf8071f8bdd6dd7eb2d6ee0da7cbe6a93670:vllm/v1/spec_decode/dflash.py",
        ],
        "summary_metrics": {
            "session_64k_kv_gib": c64k["session_kv_gib"],
            "groq_320lpu_max_64k_sessions": c64k["groq_320lpu_max_sram_sessions"],
            "wse3_2x_max_64k_sessions": c64k["wse3_2x_max_sram_sessions"],
            "gb200_8gpu_max_hbm_sessions": c64k["gb200_8gpu_max_hbm_sessions"],
            "ar_tpot_ms": a_ar["effective_tpot_ms"],
            "suffix_tpot_ms": a_suffix["effective_tpot_ms"],
            "dflash_tpot_ms": a_dflash["effective_tpot_ms"],
            "sram_hbm_cascade_tpot_ms": a_cascade["effective_tpot_ms"],
            "cascade_speedup_vs_ar_x": a_cascade["tpot_speedup_vs_ar_x"],
            "ar_cost_per_1m_usd": a_ar["cost_per_1m_tokens_usd"],
            "cascade_cost_per_1m_usd": a_cascade["cost_per_1m_tokens_usd"],
            "cascade_cost_reduction_x": round(
                a_ar["cost_per_1m_tokens_usd"] / a_cascade["cost_per_1m_tokens_usd"], 2
            ),
        },
        "sram_vs_hbm_kv_wall": kv_wall,
        "speculative_cascade_comparison": cascades,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against sealed results.json")
    args = parser.parse_args()

    computed = build_results()
    if args.check:
        if not RESULTS_PATH.exists():
            raise SystemExit(f"Missing {RESULTS_PATH}")
        expected = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if computed != expected:
            raise SystemExit("wafer_sram_gpu_cascaded_serving: results mismatch")
        print("wafer_sram_gpu_cascaded_serving: OK")
        return 0

    RESULTS_PATH.write_text(json.dumps(computed, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
