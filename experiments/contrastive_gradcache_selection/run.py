"""Synthetic InfoNCE gradient verification and transparent cache budgets."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True)
class Budget:
    local_batch: int = 128
    workers: int = 8
    dimensions: int = 768
    mini_batch: int = 8
    score_rows: int = 16
    activation_mib_per_pair: float = 64.0
    embedding_bytes: int = 2
    forward_ms_per_pair: float = 0.5
    backward_ms_per_pair: float = 1.0

    def validate(self) -> None:
        for key in ("local_batch", "workers", "dimensions", "mini_batch", "score_rows", "embedding_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.mini_batch > self.local_batch or self.score_rows > self.local_batch:
            raise ValueError("mini_batch and score_rows must not exceed local_batch")
        for key in ("activation_mib_per_pair", "forward_ms_per_pair", "backward_ms_per_pair"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")


def estimate(b: Budget) -> dict:
    b.validate()
    n = b.local_batch * b.workers
    # Conservative replicated global embeddings plus FP32 embedding gradients.
    cache_bytes = 2 * n * b.dimensions * (b.embedding_bytes + 4)
    logits_dense = 4 * b.local_batch * n
    logits_chunked = 4 * b.score_rows * n
    dense = b.local_batch * b.activation_mib_per_pair * 2**20 + logits_dense
    cached = b.mini_batch * b.activation_mib_per_pair * 2**20 + cache_bytes + logits_chunked
    monolithic_ms = b.local_batch * (b.forward_ms_per_pair + b.backward_ms_per_pair)
    cached_ms = b.local_batch * (2 * b.forward_ms_per_pair + b.backward_ms_per_pair)
    return {
        "input": asdict(b),
        "global_pairs": n,
        "global_negatives_per_query": n - 1,
        "local_negatives_per_query": b.local_batch - 1,
        "cache_and_embedding_grad_bytes": cache_bytes,
        "dense_score_bytes": logits_dense,
        "chunked_score_bytes": logits_chunked,
        "dense_modeled_bytes": dense,
        "cached_modeled_bytes": cached,
        "modeled_memory_ratio": dense / cached,
        "monolithic_compute_ms": monolithic_ms,
        "cached_compute_ms": cached_ms,
        "modeled_compute_overhead_pct": 100 * (cached_ms / monolithic_ms - 1),
        "network_latency_ms": None,
        "quality": None,
        "exclusions": "not peak HBM: model/optimizer, collective buffers, allocator and temporary softmax excluded",
    }


QUERIES = [[1.1, -0.5], [0.2, 1.3], [-0.9, 0.8], [0.6, -1.2]]
DOCUMENTS = [[1.0, -0.3], [0.4, 1.1], [-1.0, 0.7], [0.9, -0.8]]
THETA = 0.73
TEMPERATURE = 0.4


def masks(seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[0.0 if rng.random() < 0.2 else 1.25 for _ in range(2)] for _ in range(8)]


def encode(theta: float, mask: list[list[float]]) -> tuple[list, list]:
    embeddings, jacobians = [], []
    for row, keep in zip(QUERIES + DOCUMENTS, mask):
        h = [math.tanh(theta * x) for x in row]
        embeddings.append([value * m for value, m in zip(h, keep)])
        jacobians.append([(1 - value * value) * x * m for value, x, m in zip(h, row, keep)])
    return embeddings, jacobians


def infonce(embeddings: list, local_only: bool = False, detach_remote: bool = False) -> tuple[float, list]:
    n = len(embeddings) // 2
    queries, docs = embeddings[:n], embeddings[n:]
    gradients = [[0.0, 0.0] for _ in embeddings]
    loss = 0.0
    for i, query in enumerate(queries):
        candidates = [j for j in range(n) if not local_only or j // 2 == i // 2]
        logits = [sum(a*b for a, b in zip(query, docs[j])) / TEMPERATURE for j in candidates]
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        total = sum(exponentials)
        loss += (math.log(total) + maximum - logits[candidates.index(i)]) / n
        for j, exponential in zip(candidates, exponentials):
            coefficient = (exponential / total - int(i == j)) / (n * TEMPERATURE)
            for d in range(2):
                gradients[i][d] += coefficient * docs[j][d]
                if not detach_remote or j // 2 == i // 2:
                    gradients[n+j][d] += coefficient * query[d]
    return loss, gradients


def chain(embedding_gradients: list, jacobians: list, mini_batch: int = 8) -> float:
    return sum(
        sum(g * jac for g, jac in zip(embedding_gradients[i], jacobians[i]))
        for begin in range(0, len(jacobians), mini_batch)
        for i in range(begin, min(begin + mini_batch, len(jacobians)))
    )


def gradient_cases(seed: int = 3) -> dict:
    keep = masks(seed)
    embeddings, jacobians = encode(THETA, keep)
    full_loss, full_gradient = infonce(embeddings)
    epsilon = 1e-5
    plus = infonce(encode(THETA + epsilon, keep)[0])[0]
    minus = infonce(encode(THETA - epsilon, keep)[0])[0]
    reference = (plus - minus) / (2 * epsilon)
    local_loss, local_grad = infonce(embeddings, local_only=True)
    _, detached_grad = infonce(embeddings, detach_remote=True)
    different_jacobians = encode(THETA, masks(seed + 101))[1]
    raw = [
        ("global-differentiable", full_loss, chain(full_gradient, jacobians)),
        ("detached-remote-candidates", full_loss, chain(detached_grad, jacobians)),
        ("local-negative-accumulation", local_loss, chain(local_grad, jacobians)),
        ("gradcache-exact-replay", full_loss, chain(full_gradient, jacobians, mini_batch=2)),
        ("gradcache-changed-dropout", full_loss, chain(full_gradient, different_jacobians, mini_batch=2)),
    ]
    return {
        "seed": seed,
        "pairs": 4,
        "workers": 2,
        "theta": THETA,
        "temperature": TEMPERATURE,
        "finite_difference_epsilon": epsilon,
        "reference_gradient": reference,
        "tolerance": 1e-8,
        "cases": [
            {
                "id": name, "loss": loss, "gradient": gradient,
                "absolute_gradient_error": abs(gradient - reference),
                "relative_gradient_error_pct": 100 * abs(gradient - reference) / abs(reference),
                "matches_global_reference": abs(gradient - reference) < 1e-8,
                "retrieval_accuracy": None,
                "confidence": None,
            }
            for name, loss, gradient in raw
        ],
    }


def simulate() -> dict:
    base = Budget()
    return {
        "schema_version": 1,
        "round": 93,
        "provenance": "synthetic",
        "independently_reproduced": False,
        "gpu_measurements": False,
        "description": "Original scalar-encoder InfoNCE gradient checks, not a trained embedding benchmark",
        "sources": [
            {
                "url": "https://github.com/huggingface/sentence-transformers/blob/cddcaee3405a5efde83c1d98f89c6ebd6733dab6/sentence_transformers/sentence_transformer/losses/cached_multiple_negatives_ranking.py",
                "scope": "gather_across_devices=False default; differentiable gathering; chunked score rows",
            },
            {
                "url": "https://github.com/huggingface/sentence-transformers/blob/cddcaee3405a5efde83c1d98f89c6ebd6733dab6/sentence_transformers/base/losses/gradcache.py",
                "scope": "RNG snapshots, no-grad embedding pass, cached gradients and encoder replay",
            },
            {"url": "https://arxiv.org/abs/2101.06983v2", "scope": "GradCache method, not reproduced paper benchmarks"},
        ],
        "gradient_check": gradient_cases(),
        "perturbations": [gradient_cases(seed) for seed in (7, 11)],
        "budget": estimate(base),
        "scaling": [estimate(replace(base, local_batch=b)) for b in (32, 64, 128, 256)],
        "quality_status": "not measured; false negatives and task retrieval quality require datasets",
        "timing_note": "analytical assumed compute only; no communication, median, repeats or measured GPU latency",
    }


def outputs() -> dict[Path, str]:
    from .render import render_outputs
    report = simulate()
    return {
        Path(__file__).with_name("report.json"): json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        **render_outputs(report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    print("Round 93 artifacts match." if args.check else "Wrote Round 93 artifacts.")


if __name__ == "__main__":
    main()
