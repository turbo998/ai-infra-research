#!/usr/bin/env python3
"""Round 261: High-Concurrency LLM Serving Python GIL & asyncio Starvation vs ZeroMQ Multi-Process & Async Pre-Scheduling Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_serving_gil_zmq_async_schedule_tiers(
    gpu_forward_ms: float = 5.5,
) -> dict[str, Any]:
    """Evaluate 4 LLM serving CPU/GPU scheduling & process isolation architectures."""
    tiers = [
        {
            "tier_id": "tier1_monolithic_single_process_fastapi_detokenizer_scheduler_gil",
            "name": "Tier 1: 单进程大杂烩架构 (FastAPI + Detokenizer + Scheduler + GPU Runner 抢同一把 Python GIL，每步空等 14.5ms)",
            "exposed_cpu_overhead_ms": 14.5,
            "effective_itl_ms": 20.0,
            "gpu_compute_duty_cycle_pct": 27.5,
            "verdict": "GPU 跑一步只需 5.5ms，但单进程 Python 忙于拼装 128 路 SSE JSON 与分词，导致 GPU 72.5% 的时间在空转等 CPU ✕",
        },
        {
            "tier_id": "tier2_zmq_multiprocess_frontend_detokenizer_serialized_scheduler",
            "name": "Tier 2: ZeroMQ 多进程剥离 API 与 Detokenizer，但 EngineCore 内仍串行执行“等 GPU 算完 -> CPU 调度 -> 启动下一步”",
            "exposed_cpu_overhead_ms": 4.2,
            "effective_itl_ms": 9.7,
            "gpu_compute_duty_cycle_pct": 56.7,
            "verdict": "剥离了 HTTP 与分词开销，但 CPU 依然要等第 t 步 GPU 采样完拷回主机，才开始算第 t+1 步的页表与调度 ⚠️",
        },
        {
            "tier_id": "tier3_zmq_multiprocess_persistent_batch_but_sync_d2h_token_copy",
            "name": "Tier 3: 增量持久化 Batch 调度 (Persistent Batch)，但在步间同步阻塞等待 D2H 采样结果才能填下一步 input_ids",
            "exposed_cpu_overhead_ms": 2.1,
            "effective_itl_ms": 7.6,
            "gpu_compute_duty_cycle_pct": 72.4,
            "verdict": "调度器已瘦身至微秒级，但因为下一步的 input_ids 依赖上一步采样的 token.cpu()，仍在步与步之间留下 2.1ms 气泡 ✅",
        },
        {
            "tier_id": "tier4_zmq_multiprocess_plus_async_pre_scheduling_and_gpu_d2d_token_forward",
            "name": "Tier 4: ZeroMQ 多进程 + 异步前瞻调度 (Async Pre-Scheduling: 趁 GPU 算第 t 步时提前调度第 t+1 步 + GPU 显存内直传 Token)",
            "exposed_cpu_overhead_ms": 0.2,
            "effective_itl_ms": 5.7,
            "gpu_compute_duty_cycle_pct": 96.5,
            "verdict": "第 t+1 步的页表长度 (L+1) 根本不依赖第 t 步采样出哪个词！CPU 提前压入第 t+1 步 CUDA Graph，GPU 内部 D2D 直传词 ID，提速 3.51x 🏆",
        },
    ]

    return {
        "experiment_id": "llm_serving_asyncio_gil_zmq_uvloop_multiprocessing_api_selection",
        "round": 261,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "gpu_forward_ms": gpu_forward_ms,
            "batch_size": 128,
            "throughput_speedup_vs_monolithic_x": round(20.0 / 5.7, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_serving_gil_zmq_async_schedule_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
