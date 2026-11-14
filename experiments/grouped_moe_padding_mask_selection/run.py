"""Original precomputed-route ledger; no logits, expert GEMMs or native MoE."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

HERE = Path(__file__).resolve().parent
PIN = "fee9686811b7c060c404dba0c6a5a6007de4659b"
DEFAULT = {"values": [2, 3, 5, 7, 11, 13, 17, 19],
           "ids": [[0, 2], [1, 3], [0, 2], [1, 3], [0, 2], [1, 3], [0, 2], [1, 3]],
           "weights": [[3, 7] for _ in range(8)], "expert_coefficients": [2, 3, 5, 7],
           "topk": 2, "groups": 2, "padding": [False, False, True, True, False, True, True, False],
           "env_skip": True, "context": True, "aiter_fused": False, "model_optin": False,
           "consumer_skip_sentinel": True}


class LegacyCoverageError(ValueError):
    pass


def validate(p):
    if not isinstance(p, dict) or set(p) != set(DEFAULT):
        raise ValueError("exact original precomputed route and capability fields required")
    for k in ("env_skip", "context", "aiter_fused", "model_optin", "consumer_skip_sentinel"):
        if type(p[k]) is not bool:
            raise ValueError("explicit boolean witnesses required")
    for k in ("groups", "topk"):
        if type(p[k]) is not int or not 1 <= p[k] <= 32:
            raise ValueError("original groups/topk must be1..32")
    coefficients, values = p["expert_coefficients"], p["values"]
    if not isinstance(coefficients, list) or not 1 <= len(coefficients) <= 32 or any(
            type(v) is not int or not 1 <= v <= 64 for v in coefficients):
        raise ValueError("bounded positive original expert coefficients required")
    if p["topk"] > len(coefficients):
        raise ValueError("topk exceeds declared expert domain")
    if not isinstance(values, list) or len(values) > 128 or any(type(v) is not int or not 0 <= v <= 1024 for v in values):
        raise ValueError("bounded original integer row values required")
    for k in ("ids", "weights"):
        if not isinstance(p[k], list) or len(p[k]) != len(values) or any(
                not isinstance(row, list) or len(row) != p["topk"] for row in p[k]):
            raise ValueError("exact Nbytopk precomputed route arrays required")
    for row in p["ids"]:
        if any(type(v) is not int or not 0 <= v < len(coefficients) for v in row) or len(set(row)) != len(row):
            raise ValueError("unique precomputed expert ids,not negative Python indices")
    if any(type(w) is not int or not 0 <= w <= 16 for row in p["weights"] for w in row):
        raise ValueError("bounded original weight numerators required")
    mask = p["padding"]
    if mask is not None and (not isinstance(mask, list) or len(mask) > 1024 or any(type(v) is not bool for v in mask)):
        raise ValueError("padding is null or explicit boolean row witnesses")


def dispatch(values, ids, weights, coefficients):
    result = []
    calls = 0
    for value, row, weight_row in zip(values, ids, weights):
        total = 0
        for expert, weight in zip(row, weight_row):
            if expert == -1:
                if weight != 0:
                    raise ValueError("skip sentinel requires zero original weight")
                continue
            if not 0 <= expert < len(coefficients):
                raise ValueError("unsupported expert id;no negative-index fallback")
            calls += 1
            total += value * coefficients[expert] * weight
        result.append(total)
    return result, calls


def route(p, old=False):
    validate(p)
    n, experts = len(p["values"]), len(p["expert_coefficients"])
    valid_grouping = experts > p["groups"] and experts % p["groups"] == 0
    masks_padding = p["model_optin"] if old else p["model_optin"] or not p["aiter_fused"]
    mask = None
    if not valid_grouping:
        reason = "ungrouped_early_return"
    elif not masks_padding:
        reason = "router_mask_not_enabled"
    elif not p["env_skip"]:
        reason = "environment_off"
    elif not p["context"]:
        reason = "no_forward_context"
    elif p["padding"] is None:
        reason = "no_padding_witness"
    elif len(p["padding"]) < n:
        if old:
            raise LegacyCoverageError("legacy undercovered mask not executed;native outcome unknown")
        reason = "mask_undercoverage_return_none"
    else:
        mask = p["padding"][:n]
        reason = "covered_ordered_row_witness"
    if mask is not None and any(mask) and not p["consumer_skip_sentinel"]:
        raise ValueError("original consumer lacks -1skip capability;no success fallback")
    ids = [[-1 if mask is not None and mask[i] else v for v in row] for i, row in enumerate(p["ids"])]
    weights = [[0 if mask is not None and mask[i] else v for v in row] for i, row in enumerate(p["weights"])]
    values, calls = dispatch(p["values"], ids, weights, p["expert_coefficients"])
    return {"router_masks_padding_property": masks_padding, "valid_grouping": valid_grouping,
            "mask_applied": mask is not None, "reason": reason, "ids": ids, "weight_numerators": weights,
            "output_numerators": values, "original_scalar_expert_calls": calls,
            "original_weight_unit": 10, "logical_int32_fp32_route_bytes": n * p["topk"] * 8,
            "native_topk_logits_or_expert_execution_verified": None}


def paths(n):
    values = [DEFAULT["values"][i % 8] for i in range(n)]
    ids = [DEFAULT["ids"][i % 8] for i in range(n)]
    padded = [DEFAULT["padding"][i % 8] for i in range(n)]
    weights = [[0, 0] if p else [3, 7] for p in padded]
    skip_ids = [[-1, -1] if p else row for p, row in zip(padded, ids)]
    def skip():
        return dispatch(values, skip_ids, weights, DEFAULT["expert_coefficients"])
    def zero_only():
        return dispatch(values, ids, weights, DEFAULT["expert_coefficients"])
    return skip, zero_only


def measure():
    rows = []
    for n in (64, 256, 1024, 4096):
        funcs = paths(n)
        a, b = funcs[0](), funcs[1]()
        if a[0] != b[0] or (a[1], b[1]) != (n, 2*n):
            raise RuntimeError("original sentinel versus zero-weight-only dispatch differs")
        for _ in range(2):
            for f in funcs:
                f()
        samples = [[], []]
        for repeat in range(7):
            for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start = time.perf_counter_ns()
                for _ in range(3):
                    funcs[i]()
                samples[i].append((time.perf_counter_ns() - start) / 3 / 1e6)
        rows.append({"rows": n, "skip_calls": a[1], "zero_weight_calls": b[1],
                     "skip_samples_ms": samples[0], "zero_weight_samples_ms": samples[1],
                     "skip_median_ms": statistics.median(samples[0]),
                     "zero_weight_median_ms": statistics.median(samples[1])})
    return {"provenance": "measured", "python": platform.python_version(), "machine": platform.machine(),
            "warmups": 2, "repeats": 7, "operations_per_sample": 3,
            "method": "original scalar dispatch with sentinel skip versus compute-with-zero-weight",
            "excludes": "route/mask setup,logits/topk,EPLB,GEMMs,CUDA/AITER/DeepEP,capture,DPallgather,model throughput",
            "rows": rows}


def build_report(latency):
    variants = [("default_nonaiter", {}), ("env_off", {"env_skip": False}),
                ("no_context", {"context": False}), ("missing_mask", {"padding": None}),
                ("aiter_no_optin", {"aiter_fused": True}),
                ("aiter_optin", {"aiter_fused": True, "model_optin": True}),
                ("ungrouped", {"groups": 4}), ("covered_long_mask", {"padding": DEFAULT["padding"] + [True] * 8})]
    matrix = [{"name": name, "inputs": {**DEFAULT, **change}, "before": route({**DEFAULT, **change}, True),
               "after": route({**DEFAULT, **change})} for name, change in variants]
    primary = matrix[0]
    valid_rows = [i for i, pad in enumerate(DEFAULT["padding"]) if not pad]
    if any(primary["before"]["output_numerators"][i] != primary["after"]["output_numerators"][i] for i in valid_rows):
        raise RuntimeError("original live row outputs changed")
    return {"schema_version": 1, "round": 159, "provenance": "synthetic", "independently_reproduced": False,
            "source": {"repository": "vllm-project/vllm", "pin": PIN,
                       "chronology": "actualNov14UTC02:18:27/BJT10:18:27 grouped-router padding policy correction",
                       "license": "Apache-2.0 rights retained,no implementation mirror",
                       "verified": "existing env default1;grouped property optin or not AITER;context mask must cover all rows;invalid grouping returns beforemask",
                       "scope": "original precomputed id/weight ledger,not native logits/grouped-topk selection or expert/dispatch execution"},
            "case": primary, "matrix": matrix,
            "summary": {"cases": len(matrix), "changed_cases": sum(v["before"]["ids"] != v["after"]["ids"] for v in matrix),
                        "row_slots": len(DEFAULT["values"]), "live_rows": len(valid_rows),
                        "before_expert_calls": primary["before"]["original_scalar_expert_calls"],
                        "after_expert_calls": primary["after"]["original_scalar_expert_calls"],
                        "live_output_misses": sum(primary["before"]["output_numerators"][i] !=
                            primary["after"]["output_numerators"][i] for i in valid_rows),
                        "logical_route_bytes": primary["after"]["logical_int32_fp32_route_bytes"]},
            "undercoverage": {"inputs": {**DEFAULT, "model_optin": True, "padding": [False, True]},
                              "after": route({**DEFAULT, "model_optin": True, "padding": [False, True]}),
                              "native_legacy_undercovered_outcome": None},
            "latency": latency, "native_expert_flops_saved": None, "native_gpu_latency_ms": None,
            "native_allocator_peak_bytes": None, "native_deepep_v2_optin_verified": None,
            "browser_visual_acceptance": None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--measure", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.measure:
        (HERE / "latency.json").write_text(json.dumps(measure(), indent=2) + "\n", encoding="utf-8")
    r = build_report(json.loads((HERE / "latency.json").read_text()))
    from .render import outputs
    for file, text in {HERE / "report.json": json.dumps(r, indent=2) + "\n", **outputs(r)}.items():
        if args.check:
            if file.read_text(encoding="utf-8") != text:
                raise SystemExit(f"stale artifact: {file}")
        else:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
    print("Round159 sealed artifacts match" if args.check else "Wrote Round159 original artifacts")


if __name__ == "__main__":
    main()
