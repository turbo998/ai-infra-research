"""Original synthetic vocabulary-parallel CE and collective accounting."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Budget:
    tokens: int = 4096
    vocabulary: int = 131072
    partitions: int = 8
    bandwidth_gbps: float = 50.0
    alpha_us: float = 2.0

    def validate(self) -> None:
        for key in ("tokens", "vocabulary", "partitions"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.partitions > self.vocabulary:
            raise ValueError("partitions must not exceed vocabulary in this budget")
        if not math.isfinite(self.bandwidth_gbps) or self.bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be finite and positive")
        if not math.isfinite(self.alpha_us) or self.alpha_us < 0:
            raise ValueError("alpha_us must be finite and nonnegative")


def estimate(b: Budget) -> dict:
    b.validate()
    width = math.ceil(b.vocabulary / b.partitions)
    padded = width * b.partitions
    full_bytes = 4 * b.tokens * padded
    local_bytes = 4 * b.tokens * width
    allgather_send = full_bytes * (b.partitions - 1) / b.partitions
    stats_send = 3 * 4 * b.tokens * 2 * (b.partitions - 1) / b.partitions
    gather_ms = allgather_send / (b.bandwidth_gbps * 1e6) + (b.partitions - 1) * b.alpha_us / 1000
    stats_ms = stats_send / (b.bandwidth_gbps * 1e6) + 3 * 2 * (b.partitions - 1) * b.alpha_us / 1000
    return {
        "input": asdict(b),
        "padded_vocabulary": padded,
        "padding_classes": padded - b.vocabulary,
        "local_fp32_logits_bytes": local_bytes,
        "gathered_fp32_logits_bytes": full_bytes,
        "logits_storage_ratio": full_bytes / local_bytes,
        "allgather_send_bytes_per_rank": allgather_send,
        "stats_send_bytes_per_rank": stats_send,
        "forward_wire_ratio": allgather_send / stats_send if stats_send else None,
        "allgather_modeled_ms": gather_ms,
        "stats_modeled_ms": stats_ms,
        "forward_statistics_collectives": 3,
        "gpu_latency_ms": None,
        "training_quality": None,
        "scope": "FP32 logits/statistics only; excludes GEMM, hidden gradients, optimizer, workspace and backward traffic",
    }


LOGITS = [
    [1001.0, 999.0, 997.0, 1002.0, 998.0, 1003.0, 1000.0, 999.5, 996.0, 1001.5, 1010.0, 1011.0],
    [-1002.0, -1001.0, -999.0, -1003.0, -998.0, -1000.0, -1004.0, -997.0, -1001.5, -999.5, -990.0, -989.0],
    [3.0, 0.0, -1.0, 2.0, 5.0, 1.0, -2.0, 4.0, 0.5, 6.0, 10.0, 11.0],
    [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, 9.0, 10.0],
]
TARGETS = [0, 4, 9, -100]
VALID_VOCABULARY = 10
PARTITIONS = 3


def validate_loss(logits: list, targets: list[int], vocabulary: int, partitions: int) -> None:
    if not logits or len(logits) != len(targets):
        raise ValueError("logits must be nonempty and match target count")
    width = len(logits[0])
    if width == 0 or any(len(row) != width for row in logits):
        raise ValueError("logits must be a nonempty rectangular matrix")
    if type(partitions) is not int or partitions < 1 or width % partitions:
        raise ValueError("physical vocabulary must divide evenly across positive partitions")
    if type(vocabulary) is not int or not 1 <= vocabulary <= width:
        raise ValueError("invalid real vocabulary size")
    if any(type(t) is not int or (t != -100 and not 0 <= t < vocabulary) for t in targets):
        raise ValueError("target must be ignored or in the real vocabulary")
    if any(not math.isfinite(value) for row in logits for value in row):
        raise ValueError("input logits must be finite; padding is masked internally")


def dense_reference(logits: list, targets: list[int], vocabulary: int) -> dict:
    validate_loss(logits, targets, vocabulary, 1)
    valid_tokens = sum(target != -100 for target in targets)
    gradients = [[0.0] * len(logits[0]) for _ in logits]
    if not valid_tokens:
        return {"status": "no-valid-tokens: explicitly skip optimizer step", "loss": None, "gradients": gradients}
    loss = 0.0
    for i, (row, target) in enumerate(zip(logits, targets)):
        if target == -100:
            continue
        maximum = max(row[:vocabulary])
        exponentials = [math.exp(value - maximum) for value in row[:vocabulary]]
        total = sum(exponentials)
        loss += (math.log(total) - (row[target] - maximum)) / valid_tokens
        for j, value in enumerate(exponentials):
            gradients[i][j] = (value / total - int(j == target)) / valid_tokens
    return {"status": "valid", "loss": loss, "gradients": gradients}


def partitioned_loss(
    logits: list, targets: list[int], vocabulary: int, partitions: int,
    control: str = "exact", order: list[int] | None = None,
) -> dict:
    validate_loss(logits, targets, vocabulary, partitions)
    if control not in ("exact", "average-denominator", "target-on-every-shard", "unmasked-padding", "owner-local-softmax"):
        raise ValueError("unknown control")
    order = list(range(partitions)) if order is None else order
    if sorted(order) != list(range(partitions)):
        raise ValueError("order must be a permutation of all shard ids")
    width = len(logits[0]) // partitions
    valid_tokens = sum(t != -100 for t in targets)
    gradients = [[0.0] * len(logits[0]) for _ in logits]
    if not valid_tokens:
        return {"status": "no-valid-tokens: explicitly skip optimizer step", "loss": None, "gradients": gradients}
    loss = 0.0
    for i, (row, target) in enumerate(zip(logits, targets)):
        if target == -100:
            continue
        limit = len(row) if control == "unmasked-padding" else vocabulary
        shards = {
            shard: [j for j in range(shard * width, (shard + 1) * width) if j < limit]
            for shard in order
        }
        if control == "owner-local-softmax":
            shards = {target // width: shards[target // width]}
        maximum = max(max(row[j] for j in columns) for columns in shards.values() if columns)
        exponentials = {j: math.exp(row[j] - maximum) for columns in shards.values() for j in columns}
        total = sum(sum(exponentials[j] for j in columns) for columns in shards.values())
        if control == "average-denominator":
            total /= partitions
        target_shifted = sum(
            row[target] - maximum if target in columns else 0
            for columns in shards.values()
        )
        loss += (math.log(total) - target_shifted) / valid_tokens
        for j, exponential in exponentials.items():
            subtract = int(j % width == target % width) if control == "target-on-every-shard" else int(j == target)
            gradients[i][j] = (exponential / total - subtract) / valid_tokens
    return {"status": "valid", "loss": loss, "gradients": gradients}


def compare(result: dict, reference: dict) -> dict:
    error = max(abs(a - b) for r, s in zip(result["gradients"], reference["gradients"]) for a, b in zip(r, s))
    return {
        "loss": result["loss"],
        "absolute_loss_error": abs(result["loss"] - reference["loss"]),
        "maximum_gradient_error": error,
        "matches_reference": abs(result["loss"] - reference["loss"]) < 1e-10 and error < 1e-10,
        "confidence": None,
    }


def simulate() -> dict:
    reference = dense_reference(LOGITS, TARGETS, VALID_VOCABULARY)
    cases = [
        {"id": control, **compare(
            partitioned_loss(LOGITS, TARGETS, VALID_VOCABULARY, PARTITIONS, control), reference
        )}
        for control in ("exact", "average-denominator", "target-on-every-shard", "unmasked-padding", "owner-local-softmax")
    ]
    base = Budget()
    return {
        "schema_version": 1,
        "round": 94,
        "provenance": "synthetic",
        "independently_reproduced": False,
        "physical_gpu_measurement": False,
        "source": "https://github.com/NVIDIA/Megatron-LM/blob/df3a6ebe1841d9b26ab27b2cb1dcaf70b2835726/megatron/core/ops/loss/cross_entropy.py",
        "source_scope": "unsmoothed vocabulary-parallel MAX / target SUM / denominator SUM and local backward; not a padding/ignore-index API guarantee",
        "fixture": {"logits": LOGITS, "targets": TARGETS, "vocabulary": VALID_VOCABULARY, "partitions": PARTITIONS},
        "reference": {"loss": reference["loss"], "valid_tokens": 3, "tolerance": 1e-10},
        "cases": cases,
        "perturbations": [
            {"id": "reverse-shard-order", **compare(partitioned_loss(
                LOGITS, TARGETS, VALID_VOCABULARY, PARTITIONS, order=[2, 1, 0]
            ), reference)},
            {"id": "shared-logit-shift", **compare(partitioned_loss(
                [[value + 123.0 for value in row] for row in LOGITS], TARGETS, VALID_VOCABULARY, PARTITIONS
            ), reference)},
        ],
        "budget": estimate(base),
        "scaling": [estimate(replace(base, tokens=n)) for n in (1, 32, 1024, 4096)],
        "quality": None,
        "notes": [
            "Original stable analytical/finite-difference model; no PyTorch/GPU execution.",
            "Padding and ignore-index are explicit harness policies; no upstream support claim.",
            "Label smoothing, class weights and z-loss are not implemented or validated.",
            "Three sequential statistics collectives; no fused-launch or graph overlap credit.",
            "Single-rank communication ratio remains null, not zero or infinity.",
        ],
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
    print("Round 94 artifacts match." if args.check else "Wrote Round 94 artifacts.")


if __name__ == "__main__":
    main()
