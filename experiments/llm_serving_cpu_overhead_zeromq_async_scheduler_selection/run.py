"""Deterministic evaluator for LLM Serving Engine CPU Scheduling Overhead, Python GIL & Zero-Overhead Async Multiprocess Architecture (vLLM V1 / SGLang) Selection (Round 178).

Models exposed CPU scheduling & IPC overhead per decode step (ms), GPU Tensor Core active utilization (%),
P99 Decode TPOT (ms) on fast H100/B200 hardware (where pure GPU step = 5.2 ms),
total serving throughput (tokens/s), and cost per 1M output tokens (USD) across four engine CPU-frontend tiers:
  - Tier 1: Monolithic Single-Process Python Loop (vLLM V0 Style: HTTP + Tokenizer + Scheduler + ModelRunner in 1 GIL Process = 5.80ms Exposed CPU)
  - Tier 2: Multiprocess ZMQ IPC Frontend/Backend Split + Full Batch Metadata Rebuild Every Step (2.15ms Exposed CPU)
  - Tier 3: vLLM V1 Isolated EngineCore + Persistent InputBatch Delta State (`CachedRequestData` + `msgspec` Zero-Copy = 0.65ms Exposed CPU)
  - Tier 4: Zero-Overhead Overlap Scheduler (SGLang Overlap / vLLM V1 Async Output + Future Token ID Tensor on GPU = 0.08ms Exposed CPU)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_llm_serving_cpu_overhead_zeromq_async_scheduler_matrix() -> dict[str, Any]:
    workload = {
        "hardware_and_model": "8xH200 / 8xB200 Node serving 70B FP8 (Batch Size = 128 concurrent decode requests; Pure GPU forward pass = 5.20 ms/step)",
        "cpu_bottleneck_physics": "As GPUs get faster (5.2ms/step on H200/B200), a 5.8ms synchronous Python CPU scheduling + detokenization gap leaves the $300K GPU idle >52% of the time!",
        "overlap_mechanism": "CPU prepares Step N+1 metadata & page tables while GPU executes Step N; Step N+1 uses a GPU-resident Future Token ID buffer without waiting for CPU cudaStreamSynchronize()",
    }

    tiers = [
        {
            "id": "tier1_monolithic_single_process_gil_synchronous",
            "name": "Tier 1: 单进程同步主循环 (vLLM V0 架构：HTTP/分词/调度/GPU 同抢 1 个 Python GIL = 暴露 5.80ms CPU 空泡 ✕)",
            "name_en": "Tier 1: Monolithic Single-Process Loop (vLLM V0: HTTP + Tokenizer + Scheduler Share 1 GIL = 5.80ms Exposed CPU Bubble)",
            "exposed_cpu_overhead_ms": 5.80,
            "total_step_time_ms": 11.00,
            "gpu_active_utilization_pct": 47.3,
            "p99_tpot_ms": 14.8,
            "throughput_tok_s": 11636.0,
            "cost_per_1m_tokens_usd": 0.96,
            "recommended_for": "Legacy V0 prototypes only; GPU sits idle 52.7% of every decode step waiting for Python GIL, JSON parsing, and detokenization.",
            "status": "fail",
        },
        {
            "id": "tier2_multiprocess_zmq_ipc_full_state_rebuild",
            "name": "Tier 2: 多进程 ZMQ IPC 解耦 HTTP/Tokenizer + 每步全量重建张量元数据 (暴露 2.15ms CPU 空泡)",
            "name_en": "Tier 2: Multiprocess ZMQ IPC Frontend Split + Per-Step Full Metadata Rebuild (2.15ms Exposed CPU Bubble)",
            "exposed_cpu_overhead_ms": 2.15,
            "total_step_time_ms": 7.35,
            "gpu_active_utilization_pct": 70.7,
            "p99_tpot_ms": 9.6,
            "throughput_tok_s": 17415.0,
            "cost_per_1m_tokens_usd": 0.64,
            "recommended_for": "Intermediate multi-process engines; eliminates HTTP/Tokenizer GIL contention but still stalls GPU on synchronous sampler sync & full block-table copies.",
            "status": "warn",
        },
        {
            "id": "tier3_vllm_v1_persistent_batch_delta_msgspec",
            "name": "Tier 3: vLLM V1 独立 EngineCore + 持久化增量批状态 (`CachedRequestData` 仅传增量 = 暴露 0.65ms)",
            "name_en": "Tier 3: vLLM V1 Isolated EngineCore + Persistent Batch Delta State (`CachedRequestData` + `msgspec` = 0.65ms Exposed)",
            "exposed_cpu_overhead_ms": 0.65,
            "total_step_time_ms": 5.85,
            "gpu_active_utilization_pct": 88.9,
            "p99_tpot_ms": 7.2,
            "throughput_tok_s": 21880.0,
            "cost_per_1m_tokens_usd": 0.51,
            "recommended_for": "Standard vLLM V1 production deployments: caches request page tables across steps in GPU worker memory and transmits only 1-token deltas.",
            "status": "pass",
        },
        {
            "id": "tier4_zero_overhead_overlap_scheduler_future_token",
            "name": "Tier 4: 零开销异步重叠调度器 (SGLang Overlap / vLLM V1 Async Output + GPU Future Token 槽位 = 仅 0.08ms 🏆)",
            "name_en": "Tier 4: Zero-Overhead Overlap Scheduler (SGLang Overlap / vLLM V1 Async Output + GPU Future Token Slot = 0.08ms Exposed)",
            "exposed_cpu_overhead_ms": 0.08,
            "total_step_time_ms": 5.28,
            "gpu_active_utilization_pct": 98.5,
            "p99_tpot_ms": 5.8,
            "throughput_tok_s": 24242.0,
            "cost_per_1m_tokens_usd": 0.46,
            "recommended_for": "High-throughput H200/B200 serving & speculative decoding: hides 98.6% of CPU scheduling behind Step N GPU execution (108% higher throughput vs V0).",
            "status": "pass",
        },
    ]

    t1, t2, t3, t4 = tiers
    summary = {
        "tier4_vs_tier1_cpu_overhead_reduction_ratio": round(
            t1["exposed_cpu_overhead_ms"] / t4["exposed_cpu_overhead_ms"], 1
        ),
        "tier4_vs_tier1_throughput_speedup_ratio": round(
            t4["throughput_tok_s"] / t1["throughput_tok_s"], 2
        ),
        "tier4_vs_tier1_gpu_util_gain_pts": round(
            t4["gpu_active_utilization_pct"] - t1["gpu_active_utilization_pct"], 1
        ),
        "tier4_token_cost_reduction_pct": round(
            (t1["cost_per_1m_tokens_usd"] - t4["cost_per_1m_tokens_usd"]) / t1["cost_per_1m_tokens_usd"] * 100.0, 1
        ),
    }

    return {
        "experiment_id": "llm_serving_cpu_overhead_zeromq_async_scheduler_selection",
        "date": "2026-11-15",
        "round": 178,
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_published_engine_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "vLLM V1 Architecture Whitepaper (Isolated EngineCore, ZeroMQ IPC, Persistent InputBatch & CachedRequestData Delta)",
                "SGLang Zero-Overhead Batch Scheduler (Overlap Scheduler with Future Token Map & Dual CUDA Streams)",
                "Nsight Systems CPU-GPU Timeline Traces on 8xH100 / 8xH200 Running 70B FP8 Decode Steps",
            ],
        },
        "workload": workload,
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_llm_serving_cpu_overhead_zeromq_async_scheduler_matrix()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
