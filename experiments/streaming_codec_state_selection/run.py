"""Original causal codec-state toy: aligned convolutions and overlap-add."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path


def validate_signal(values: list[float]) -> None:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("signal must be nonempty and finite")


def validate_kernel(weights: list[float], stride: int, dilation: int = 1) -> None:
    validate_signal(weights)
    if type(stride) is not int or type(dilation) is not int or stride < 1 or dilation < 1:
        raise ValueError("stride and dilation must be positive integers")
    if (len(weights) - 1) * dilation + 1 < stride:
        raise ValueError("effective kernel must cover the stride")


def offline_causal(signal: list, weights: list, stride: int, bias: float = 0.0, dilation: int = 1) -> list:
    validate_signal(signal)
    validate_kernel(weights, stride, dilation)
    history = (len(weights) - 1) * dilation + 1 - stride
    result = []
    for frame in range(math.ceil(len(signal) / stride)):
        value = bias
        for tap, weight in enumerate(weights):
            position = frame * stride - history + tap * dilation
            value += weight * (signal[position] if 0 <= position < len(signal) else 0)
        result.append(value)
    return result


class CausalConv:
    def __init__(self, weights: list, stride: int, bias: float = 0.0, dilation: int = 1):
        validate_kernel(weights, stride, dilation)
        if not math.isfinite(bias):
            raise ValueError("bias must be finite")
        self.weights = list(weights)
        self.stride = stride
        self.bias = bias
        self.dilation = dilation
        self.history_size = (len(weights) - 1) * dilation + 1 - stride
        self.reset()

    def reset(self):
        self.previous = [0.0] * self.history_size

    def step(self, values: list) -> list:
        validate_signal(values)
        if len(values) % self.stride:
            raise ValueError("kernel steps must be positive multiples of stride")
        joined = self.previous + list(values)
        output = [
            self.bias + sum(weight * joined[offset + tap * self.dilation] for tap, weight in enumerate(self.weights))
            for offset in range(0, len(values), self.stride)
        ]
        self.previous = joined[-self.history_size:] if self.history_size else []
        return output


class AlignedIngress:
    """Transport buffering is a harness adapter, not an upstream API claim."""

    def __init__(self, conv: CausalConv, reset_each_chunk: bool = False):
        self.conv = conv
        self.reset_each_chunk = reset_each_chunk
        self.pending = []
        self.closed = False
        self.eos_padding = 0

    def reset(self):
        self.conv.reset()
        self.pending = []
        self.closed = False
        self.eos_padding = 0

    def feed(self, values: list) -> list:
        if self.closed:
            raise ValueError("stream is closed; create or reset its state before reuse")
        validate_signal(values)
        self.pending.extend(values)
        count = len(self.pending) // self.conv.stride * self.conv.stride
        if not count:
            return []
        aligned, self.pending = self.pending[:count], self.pending[count:]
        if self.reset_each_chunk:
            self.conv.reset()
        return self.conv.step(aligned)

    def finish(self, policy: str = "pad") -> list:
        if self.closed:
            raise ValueError("stream has already finished")
        if policy not in ("pad", "reject"):
            raise ValueError("EOS policy must be pad or reject")
        if self.pending and policy == "reject":
            raise ValueError("incomplete stride frame at EOS")
        output = []
        if self.pending:
            self.eos_padding = self.conv.stride - len(self.pending)
            output = self.feed([0.0] * self.eos_padding)
        self.closed = True
        return output


def offline_transpose(frames: list, weights: list, stride: int, bias: float) -> list:
    validate_signal(frames)
    validate_kernel(weights, stride)
    output = [bias] * (len(frames) * stride + len(weights) - stride)
    for position, frame in enumerate(frames):
        for tap, weight in enumerate(weights):
            output[position * stride + tap] += frame * weight
    return output[:len(frames) * stride]


class TransposeConv:
    def __init__(self, weights: list, stride: int, bias: float, duplicate_bias: bool = False):
        validate_kernel(weights, stride)
        if not math.isfinite(bias):
            raise ValueError("bias must be finite")
        self.weights, self.stride, self.bias = list(weights), stride, bias
        self.duplicate_bias = duplicate_bias
        self.reset()

    def reset(self):
        self.partial = [0.0] * (len(self.weights) - self.stride)

    def step(self, frames: list) -> list:
        validate_signal(frames)
        size = len(frames) * self.stride
        output = [self.bias] * (size + len(self.partial))
        for position, frame in enumerate(frames):
            for tap, weight in enumerate(self.weights):
                output[position * self.stride + tap] += frame * weight
        for j, value in enumerate(self.partial):
            output[j] += value
        if self.partial:
            self.partial = [
                value - (0.0 if self.duplicate_bias else self.bias)
                for value in output[size:]
            ]
        return output[:size]


@dataclass(frozen=True)
class Budget:
    streams: int = 32
    channels: int = 64
    kernel: int = 7
    stride: int = 4
    dilation: int = 1
    scalar_bytes: int = 2
    stage_rate_hz: int = 24000
    chunk_samples: int = 960
    baseline_samples: int = 24000


def estimate(b: Budget) -> dict:
    for name, value in asdict(b).items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if b.kernel < b.stride:
        raise ValueError("transpose kernel must be at least its stride")
    if b.scalar_bytes not in (2, 4):
        raise ValueError("scalar_bytes must be 2 or 4")
    if b.chunk_samples % b.stride:
        raise ValueError("processing chunk must divide into full stride frames")
    effective = (b.kernel - 1) * b.dilation + 1
    factor = b.streams * b.channels * b.scalar_bytes
    conv = factor * (effective - b.stride)
    transpose = factor * (b.kernel - b.stride)
    ingress = factor * (b.stride - 1)
    return {
        "input": asdict(b),
        "conv_history_bytes": conv,
        "transpose_partial_bytes": transpose,
        "ingress_pending_bound_bytes": ingress,
        "total_state_bound_bytes": conv + transpose + ingress,
        "chunk_collection_bound_ms": 1000 * b.chunk_samples / b.stage_rate_hz,
        "baseline_collection_bound_ms": 1000 * b.baseline_samples / b.stage_rate_hz,
        "gpu_latency_ms": None,
        "audio_quality": None,
        "scope": "one illustrative input/output-channel-matched convolution pair plus ingress; no complete codec peak-memory claim",
    }


SIGNAL = [-0.5, 0.2, 0.9, -0.3, 0.1, 0.8, -0.6, 0.4, 0.7, -0.2, 0.3, -0.9, 0.5, 0.6, -0.1, 0.0, 0.25]
WEIGHTS = [0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.25]
FRAME_SIGNAL = [0.2, -0.5, 1.3, 0.0, 0.7, -0.9]


def chunks(signal: list, lengths: list[int]) -> list[list]:
    if any(type(n) is not int or n < 1 for n in lengths) or sum(lengths) != len(signal):
        raise ValueError("positive chunk lengths must cover the entire signal")
    output, offset = [], 0
    for count in lengths:
        output.append(signal[offset:offset+count])
        offset += count
    return output


def stream_causal(lengths: list[int], reset_each: bool = False) -> tuple[list, int]:
    ingress = AlignedIngress(CausalConv(WEIGHTS, 4, 0.125), reset_each)
    output = []
    for chunk in chunks(SIGNAL, lengths):
        output.extend(ingress.feed(chunk))
    output.extend(ingress.finish())
    return output, ingress.eos_padding


def comparison(name: str, actual: list, expected: list) -> dict:
    same_length = len(actual) == len(expected)
    error = max(abs(a-b) for a,b in zip(actual, expected)) if same_length else None
    return {
        "id": name, "output": actual, "reference_output": expected,
        "output_samples": len(actual), "reference_samples": len(expected),
        "same_length": same_length, "maximum_absolute_error": error,
        "matches_reference": same_length and error < 1e-12,
        "confidence": None,
    }


def simulate() -> dict:
    causal_reference = offline_causal(SIGNAL, WEIGHTS, 4, 0.125)
    lengths = [1, 2, 5, 3, 6]
    exact, pad = stream_causal(lengths)
    wrong_reset, _ = stream_causal(lengths, True)
    dropping, drop_conv = [], CausalConv(WEIGHTS, 4, 0.125)
    for chunk in chunks(SIGNAL, lengths):
        usable = len(chunk) // 4 * 4
        if usable:
            dropping.extend(drop_conv.step(chunk[:usable]))
    transpose_reference = offline_transpose(FRAME_SIGNAL, WEIGHTS, 4, 0.4)
    transpose_outputs = []
    for duplicate in (False, True):
        conv, result = TransposeConv(WEIGHTS, 4, 0.4, duplicate), []
        for chunk in chunks(FRAME_SIGNAL, [1, 2, 1, 2]):
            result.extend(conv.step(chunk))
        transpose_outputs.append(result)
    next_request = [1.0, 0.0, -1.0, 0.2, 0.4, -0.3, 0.1, 0.7]
    reused = CausalConv(WEIGHTS, 4, 0.125)
    reused.step(SIGNAL[:12])
    stale = reused.step(next_request)
    reused.reset()
    reset_output = reused.step(next_request)
    fresh_reference = offline_causal(next_request, WEIGHTS, 4, 0.125)
    cases = [
        comparison("aligned-ingress-and-EOS-padding", exact, causal_reference),
        comparison("reset-history-every-chunk", wrong_reset, causal_reference),
        comparison("drop-transport-remainders", dropping, causal_reference),
        comparison("transpose-bias-free-carry", transpose_outputs[0], transpose_reference),
        comparison("transpose-carry-includes-bias", transpose_outputs[1], transpose_reference),
        comparison("reuse-slot-with-stale-history", stale, fresh_reference),
        comparison("reset-slot-before-new-request", reset_output, fresh_reference),
    ]
    base = Budget()
    return {
        "schema_version": 1, "round": 95, "provenance": "synthetic",
        "independently_reproduced": False, "physical_gpu_measurement": False,
        "source": "https://github.com/kyutai-labs/moshi/blob/fd014fc2d2e2f18b809c41d2da11dbc3817efcdf/moshi/moshi/modules/conv.py",
        "source_scope": "constant causal padding, positive stride-aligned steps, history/partial reset and bias-free transpose carry",
        "fixture": {"input": SIGNAL, "weights": WEIGHTS, "stride": 4, "bias": 0.125,
                    "ingress_chunk_lengths": lengths, "eos_padding_samples": pad, "tolerance": 1e-12},
        "cases": cases,
        "perturbations": [
            comparison(str(pattern), stream_causal(pattern)[0], causal_reference)
            for pattern in ([17], [4, 4, 4, 5], [1] * 17)
        ],
        "budget": estimate(base),
        "scaling": [estimate(replace(base, chunk_samples=count)) for count in (240, 480, 960, 1920)],
        "quality": None,
        "notes": [
            "Original scalar model, not an execution of PyTorch or Mimi.",
            "Arbitrary ingress and right-zero EOS padding are explicit adapter policies, not upstream API promises.",
            "Transpose retains exactly K-S future contributions and trims the final tail, matching the chosen causal reference.",
            "History is effective_kernel-stride; this alignment differs from the common kernel-1 convention.",
            "Collection bounds assume a continuous stage-rate input, excluding network, computation and playout.",
            "No audio quality, subjective listening score, GPU latency or service P99 is inferred.",
        ],
    }


def outputs() -> dict[Path,str]:
    from .render import render_outputs
    report = simulate()
    return {Path(__file__).with_name("report.json"): json.dumps(report, indent=2) + "\n", **render_outputs(report)}


def main():
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
    print("Round 95 artifacts match." if args.check else "Wrote Round 95 artifacts.")


if __name__ == "__main__":
    main()
