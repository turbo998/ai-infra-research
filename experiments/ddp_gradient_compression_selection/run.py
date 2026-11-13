"""Synthetic communication accounting and a small distributed rank-one model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True)
class Profile:
    rows: int = 4096
    columns: int = 4096
    matrices: int = 16
    vector_elements: int = 65536
    workers: int = 64
    rank: int = 4
    min_compression_rate: float = 2.0
    bandwidth_gbps: float = 25.0
    alpha_us: float = 2.0
    overlap_ms: float = 20.0
    backward_ms: float = 100.0
    codec_ms: float = 8.0
    cast_ms: float = 1.0

    def validate(self) -> None:
        for name in ("rows", "columns", "matrices", "workers", "rank"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.vector_elements) is not int or self.vector_elements < 0:
            raise ValueError("vector_elements must be a nonnegative integer")
        if self.rank > min(self.rows, self.columns):
            raise ValueError("rank must not exceed the smaller matrix dimension")
        for name in ("min_compression_rate", "bandwidth_gbps", "backward_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("alpha_us", "overlap_ms", "codec_ms", "cast_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.overlap_ms > self.backward_ms:
            raise ValueError("overlap_ms must not exceed backward_ms")


def estimate(profile: Profile) -> dict:
    """One aggregate bucket; decimal GB/s; ideal ring, not an NCCL prediction."""
    profile.validate()
    p = profile
    dense_elements = p.rows * p.columns * p.matrices + p.vector_elements
    factor_elements = (p.rows + p.columns) * p.rank * p.matrices
    compress = (
        p.rows > 1
        and p.columns > 1
        and (p.rows + p.columns) * p.rank * p.min_compression_rate
        < p.rows * p.columns
    )
    compressed_bytes = 4 * (factor_elements + p.vector_elements)
    phases = 2 + int(p.vector_elements > 0) if compress else 1
    ring_factor = 2 * (p.workers - 1) / p.workers

    def path(payload: int, collectives: int, codec: float) -> dict:
        wire = payload * ring_factor
        communication = (
            wire / (p.bandwidth_gbps * 1_000_000)
            + collectives * 2 * (p.workers - 1) * p.alpha_us / 1000
        )
        service = communication + codec
        exposed = max(0.0, service - p.overlap_ms)
        return {
            "payload_bytes": payload,
            "ring_send_bytes_per_rank": wire,
            "collectives": collectives,
            "communication_ms": communication,
            "codec_ms": codec,
            "service_ms": service,
            "exposed_ms": exposed,
            "step_ms": p.backward_ms + exposed,
        }

    dense = path(4 * dense_elements, 1, 0)
    bf16 = path(2 * dense_elements, 1, p.cast_ms)
    power = path(
        compressed_bytes if compress else 4 * dense_elements,
        phases,
        p.codec_ms if compress else 0,
    )
    winner = min(
        (("dense", dense), ("bf16", bf16), ("powersgd", power)),
        key=lambda item: item[1]["step_ms"],
    )[0]
    return {
        "input": asdict(p),
        "compression_eligible": compress,
        "dense": dense,
        "bf16": bf16,
        "powersgd": power,
        "payload_compression_ratio": dense["payload_bytes"] / power["payload_bytes"],
        "powersgd_step_speedup": dense["step_ms"] / power["step_ms"],
        "powersgd_saved_ms": dense["step_ms"] - power["step_ms"],
        "timing_only_winner": winner,
        "quality_gate": "pending: train to matched validation quality",
        "training_quality": None,
        "confidence": None,
    }


def norm(values: list[float]) -> float:
    return math.sqrt(sum(x * x for x in values))


def rank_one_step(
    gradients: list[list[list[float]]],
    residuals: list[list[list[float]]],
    q: list[float],
    error_feedback: bool,
) -> tuple[list[list[float]], list[list[list[float]]], list[float]]:
    """Two factor means with shared Q; no independent local SVDs."""
    workers, rows, columns = len(gradients), len(gradients[0]), len(q)
    corrected = [
        [
            [
                gradients[w][i][j] + (residuals[w][i][j] if error_feedback else 0)
                for j in range(columns)
            ]
            for i in range(rows)
        ]
        for w in range(workers)
    ]
    q_norm = norm(q)
    if q_norm == 0:
        raise ValueError("shared Q must be nonzero")
    q = [x / q_norm for x in q]
    p = [
        sum(sum(matrix[i][j] * q[j] for j in range(columns)) for matrix in corrected)
        / workers
        for i in range(rows)
    ]
    p_norm = norm(p)
    p = [x / p_norm for x in p] if p_norm else [0.0] * rows
    next_q = [
        sum(sum(matrix[i][j] * p[i] for i in range(rows)) for matrix in corrected)
        / workers
        for j in range(columns)
    ]
    recovered = [[p[i] * next_q[j] for j in range(columns)] for i in range(rows)]
    new_residuals = [
        [
            [matrix[i][j] - recovered[i][j] for j in range(columns)]
            for i in range(rows)
        ]
        for matrix in corrected
    ]
    # A zero projection must not strand future gradients in a zero warm-start.
    return recovered, new_residuals, next_q if norm(next_q) else q


def toy_feedback(seed: int = 7, steps: int = 24, error_feedback: bool = True) -> dict:
    rng = random.Random(seed)
    workers, size = 3, 4
    residuals = [[[0.0] * size for _ in range(size)] for _ in range(workers)]
    q = [rng.uniform(-1, 1) for _ in range(size)]
    cumulative = [[0.0] * size for _ in range(size)]
    truth = [[0.0] * size for _ in range(size)]
    for step in range(steps):
        gradients = [
            [
                [
                    (4.0 / (i + 1) if i == j else 0)
                    + 0.03 * (w - 1)
                    + 0.02 * math.sin(step + i + 2 * j)
                    for j in range(size)
                ]
                for i in range(size)
            ]
            for w in range(workers)
        ]
        recovered, residuals, q = rank_one_step(gradients, residuals, q, error_feedback)
        for i in range(size):
            for j in range(size):
                cumulative[i][j] += recovered[i][j]
                truth[i][j] += sum(g[i][j] for g in gradients) / workers
    difference = [truth[i][j] - cumulative[i][j] for i in range(size) for j in range(size)]
    mean_residual = [
        sum(r[i][j] for r in residuals) / workers
        for i in range(size)
        for j in range(size)
    ]
    return {
        "seed": seed,
        "steps": steps,
        "workers": workers,
        "matrix_shape": [size, size],
        "rank": 1,
        "error_feedback": error_feedback,
        "cumulative_relative_error_pct": 100 * norm(difference)
        / norm([v for row in truth for v in row]),
        "telescoping_residual_gap": norm(
            [a - b for a, b in zip(difference, mean_residual)]
        ),
        "final_mean_residual_norm": norm(mean_residual),
        "validation_loss": None,
    }


def simulate() -> dict:
    base = Profile()
    cases = [
        ("bandwidth-bound", base, "powersgd"),
        ("fast-fabric-hidden", replace(base, bandwidth_gbps=200), "dense"),
        ("small-matrices", replace(base, rows=8, columns=8, matrices=1), "dense"),
        ("high-rank-fallback", replace(base, rank=1024), "bf16"),
        ("vector-dominated", replace(base, vector_elements=536870912), "bf16"),
        ("no-overlap", replace(base, overlap_ms=0), "powersgd"),
    ]
    scenarios = [
        {"id": name, "expected_timing_winner": expected, **estimate(profile)}
        for name, profile, expected in cases
    ]
    robustness = []
    for bandwidth in (10, 25, 100, 200):
        row = estimate(replace(base, bandwidth_gbps=bandwidth))
        robustness.append({
            "bandwidth_gbps": bandwidth,
            "dense_ms": row["dense"]["step_ms"],
            "powersgd_ms": row["powersgd"]["step_ms"],
            "saved_ms": row["powersgd_saved_ms"],
        })
    feedback = [
        toy_feedback(seed=seed, error_feedback=enabled)
        for seed in (7, 11)
        for enabled in (False, True)
    ]
    return {
        "schema_version": 1,
        "round": 92,
        "provenance": "synthetic",
        "independently_reproduced": False,
        "physical_gpu_measurement": False,
        "hardware": "Python standard-library analytical model; no GPU or API",
        "latency_statistic": "one deterministic evaluation; not measured; no median",
        "sources": [
            {
                "id": "pytorch-v2.10.0-default-hooks",
                "url": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py",
                "verified_on": "2026-11-13",
                "kind": "implementation",
            },
            {
                "id": "pytorch-v2.10.0-powersgd",
                "url": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/distributed/algorithms/ddp_comm_hooks/powerSGD_hook.py",
                "verified_on": "2026-11-13",
                "kind": "implementation",
            },
            {
                "id": "pytorch-v2.10-ddp-hooks",
                "url": "https://docs.pytorch.org/docs/2.10/ddp_comm_hooks.html",
                "verified_on": "2026-11-13",
                "kind": "documentation",
            },
            {
                "id": "sglang-consortium-20261112",
                "url": "https://www.lmsys.org/blog/2026-11-12-sglang-consortium",
                "verified_on": "2026-11-13",
                "kind": "daily-context-not-compression-evidence",
            },
        ],
        "assumptions": [
            "FP32 dense gradients and factors; BF16 wire cast is a separate path.",
            "One aggregate matrix group plus optional dense vectors per modeled bucket.",
            "Ideal ring: 2(W-1)/W payload and 2(W-1) alpha per collective.",
            "P and Q are sequential; optional vector reduction is another phase.",
            "Codec and communication share one hideable overlap window.",
            "Codec costs and effective bandwidth are user assumptions, not measurements.",
            "Residual storage is one FP32 gradient buffer; peak workspace is not modeled.",
            "No claim about convergence, FP8 scaling, FSDP, NCCL algorithms or production.",
        ],
        "scenarios": scenarios,
        "robustness": robustness,
        "feedback": feedback,
        "summary": {
            "scenario_count": len(scenarios),
            "timing_policy_matches": sum(
                row["timing_only_winner"] == row["expected_timing_winner"] for row in scenarios
            ),
            "production_approved": 0,
            "quality_status": "not measured",
        },
    }


def outputs() -> dict[Path, str]:
    from .render import render_outputs

    report = simulate()
    here = Path(__file__).resolve().parent
    return {
        here / "report.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        **render_outputs(report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare sealed artifacts without writing")
    args = parser.parse_args()
    for path, content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    print("Round 92 synthetic artifacts match." if args.check else "Wrote Round 92 synthetic artifacts.")


if __name__ == "__main__":
    main()
