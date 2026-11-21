from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "cuda_graph_piecewise_torch_compile_vllm_v1_memory_pool_selection",
        "title": "CUDA Graph & Dynamic Compilation Selection: Piecewise CUDA Graphs (vLLM V1 + torch.compile) vs. Full Decode Graphs & Eager Mode",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "model": "Llama-3.3-70B-FP8 (80 Layers, Mixed Chunked-Prefill + Decode Workload)",
            "hardware": "2x NVIDIA H100 80GB (TP=2)",
            "captured_shape_buckets": 32,
        },
        "tiers": [
            {
                "tier": "Tier 1: PyTorch Eager Mode (--enforce-eager, No CUDA Graphs)",
                "small_batch_tpot_ms": 14.8,
                "mixed_chunked_prefill_decode_step_ms": 22.5,
                "cuda_graph_memory_pool_gb_per_gpu": 0.0,
                "warm_startup_compile_sec": 0.0,
                "recommended_scenario": "Debugging only (pays 3.2ms CPU kernel launch overhead on 320+ kernels/step)",
            },
            {
                "tier": "Tier 2: Legacy Full-Model Decode-Only CUDA Graphs (vLLM V0, 35 Batch Buckets)",
                "small_batch_tpot_ms": 10.6,
                "mixed_chunked_prefill_decode_step_ms": 21.8,
                "cuda_graph_memory_pool_gb_per_gpu": 4.8,
                "warm_startup_compile_sec": 24.0,
                "recommended_scenario": "Legacy V0 engines (falls back to slow Eager Mode whenever a Chunked Prefill joins the batch!)",
            },
            {
                "tier": "Tier 3: Uncached torch.compile + Full CUDA Graphs (Cold Inductor Compilation Every Boot)",
                "small_batch_tpot_ms": 9.4,
                "mixed_chunked_prefill_decode_step_ms": 20.5,
                "cuda_graph_memory_pool_gb_per_gpu": 4.2,
                "warm_startup_compile_sec": 145.0,
                "recommended_scenario": "Anti-pattern in autoscaling K8s (145s cold Triton/Inductor compilation stalls pod readiness)",
            },
            {
                "tier": "Tier 4 (Recommended): vLLM V1 Piecewise CUDA Graphs + torch.compile Fusion + Persistent Disk Cache",
                "small_batch_tpot_ms": 8.9,
                "mixed_chunked_prefill_decode_step_ms": 14.2,
                "cuda_graph_memory_pool_gb_per_gpu": 1.5,
                "warm_startup_compile_sec": 3.8,
                "recommended_scenario": "Production vLLM V1 / SGLang accelerating BOTH Decode & Mixed Chunked-Prefill (-68.8% Graph HBM)",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
