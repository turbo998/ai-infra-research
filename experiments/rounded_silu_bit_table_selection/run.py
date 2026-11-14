"""Original finite-word arithmetic/cache model, not CUDA/libdevice parity."""

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import time

HERE = Path(__file__).resolve().parent
PIN = "6a64408419ccc4a091e3982c5b9fc10c5bda0f0e"
DTYPES = ("float16", "bfloat16")
TABLE_BYTES = 65536 * 2
DEFAULT = {"dtype": "float16", "device": 0, "cuda": True, "hip": False,
           "major": 10, "shared_optin_bytes": TABLE_BYTES, "capturing": False,
           "cached_keys": [], "operation": "silu", "clamp": False, "tokens": 1}
GATES = {"float16": [0, 0x8000, 0x0401, 0x0403, 0x0406, 0x3C00, 0xBC00, 0xD590],
         "bfloat16": [0, 0x8000, 0x0081, 0x0083, 0x0085, 0x3F80, 0xBF80, 0xC2B2]}


def f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def decode(word, dtype):
    if dtype not in DTYPES or type(word) is not int or not 0 <= word <= 65535:
        raise ValueError("original uint16 word and float16/bfloat16 dtype required")
    if dtype == "float16":
        return struct.unpack("<e", word.to_bytes(2, "little"))[0]
    return struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0]


def encode(value, dtype):
    if dtype not in DTYPES or math.isnan(value):
        raise ValueError("original prototype does not certify NaN payloads")
    if dtype == "float16":
        value = math.copysign(math.inf, value) if abs(value) >= 65520 else value
        return int.from_bytes(struct.pack("<e", value), "little")
    raw = int.from_bytes(struct.pack("<f", f32(value)), "little")
    return ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16) & 65535


def activation(x):
    exp = math.inf if -x > 90 else f32(math.exp(-x))
    return f32(x / f32(1 + exp))


def finite(word, dtype):
    x = decode(word, dtype)
    if not math.isfinite(x):
        raise ValueError("finite input words only; native infinity/NaN payload parity unmeasured")
    return x


def lane(gate, up, dtype):
    g, u = finite(gate, dtype), finite(up, dtype)
    a = activation(g)
    rounded = encode(a, dtype)
    return {"gate_word": gate, "up_word": up, "activation_word": rounded,
            "rounded_output_word": encode(f32(decode(rounded, dtype) * u), dtype),
            "single_round_output_word": encode(f32(a * u), dtype)}


def sparse_table(gates, dtype):
    return {g: encode(activation(finite(g, dtype)), dtype) for g in gates}


def lookup(gate, up, dtype, table):
    finite(gate, dtype)
    u = finite(up, dtype)
    if gate not in table:
        raise ValueError("word absent from original sparse fixture; no zero/default activation")
    return encode(f32(decode(table[gate], dtype) * u), dtype)


def direct_output(gate, up, dtype):
    g, u = finite(gate, dtype), finite(up, dtype)
    a = encode(activation(g), dtype)
    return encode(f32(decode(a, dtype) * u), dtype)


def route(p):
    if not isinstance(p, dict) or set(p) != set(DEFAULT):
        raise ValueError("exact original dispatch/cache fields required")
    if not isinstance(p["dtype"], str) or p["dtype"] not in (*DTYPES, "float32"):
        raise ValueError("explicit known dtype required")
    if p["operation"] not in ("silu", "gelu", "gelu_tanh"):
        raise ValueError("explicit known activation required")
    for key in ("cuda", "hip", "capturing", "clamp"):
        if type(p[key]) is not bool:
            raise ValueError("explicit booleans required")
    for key in ("device", "major", "shared_optin_bytes", "tokens"):
        if type(p[key]) is not int or not 0 <= p[key] <= 2**32 - 1:
            raise ValueError("bounded nonnegative integer dispatch fields required")
    keys = p["cached_keys"]
    if not isinstance(keys, list) or any(not isinstance(k, list) or len(k) != 2
            or k[0] not in DTYPES or type(k[1]) is not int or not 0 <= k[1] <= 2**32 - 1 for k in keys):
        raise ValueError("cache keys are [16-bit dtype, device index] pairs")
    if len({tuple(k) for k in keys}) != len(keys):
        raise ValueError("duplicate original cache keys")
    key = [p["dtype"], p["device"]]
    eligible = p["operation"] == "silu" and not p["clamp"] and p["dtype"] in DTYPES and p["cuda"]
    cached = eligible and key in keys
    supported = not p["hip"] and p["major"] == 10 and p["shared_optin_bytes"] >= TABLE_BYTES
    if not eligible:
        action, reason = "fallback", "activation_clamp_dtype_or_device"
    elif cached:
        action, reason = "reuse", "exact_dtype_device_key_before_capture_or_property_probe"
    elif not supported:
        action, reason = "fallback", "uncached_hardware_gate"
    elif p["capturing"]:
        action, reason = "fallback", "cold_capture_must_not_allocate"
    else:
        action, reason = "build", "uncaptured_supported_cache_miss"
    after = [k[:] for k in keys]
    if action == "build":
        after.append(key)
    selected = action in ("build", "reuse")
    return {"action": action, "reason": reason, "table_key": key if eligible else None,
            "cached_keys_after": after, "new_logical_table_bytes": TABLE_BYTES if action == "build" else 0,
            "table_selected": selected, "activation_launch": p["tokens"] > 0,
            "shared_bytes_per_table_cta": TABLE_BYTES if selected and p["tokens"] > 0 else 0,
            "native_driver_or_kernel_execution_verified": None}


def pairs(dtype):
    up = encode(1.5, dtype)
    return [(g, up) for g in GATES[dtype]]


def paths(n, dtype="float16"):
    inputs = [pairs(dtype)[i % len(GATES[dtype])] for i in range(n)]
    table = sparse_table(GATES[dtype], dtype)
    def lut():
        return [lookup(g, u, dtype, table) for g, u in inputs]
    def direct():
        return [direct_output(g, u, dtype) for g, u in inputs]
    return lut, direct


def measure():
    rows = []
    for n in (64, 256, 1024, 4096):
        funcs = paths(n)
        if funcs[0]() != funcs[1]():
            raise RuntimeError("original sparse lookup and rounded arithmetic differ")
        for _ in range(2):
            for func in funcs:
                func()
        samples = [[], []]
        for repeat in range(7):
            for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start = time.perf_counter_ns()
                for _ in range(3):
                    funcs[i]()
                samples[i].append((time.perf_counter_ns() - start) / 3 / 1e6)
        rows.append({"lanes": n, "lookup_samples_ms": samples[0], "direct_samples_ms": samples[1],
                     "lookup_median_ms": statistics.median(samples[0]),
                     "direct_median_ms": statistics.median(samples[1])})
    return {"provenance": "measured", "python": platform.python_version(), "machine": platform.machine(),
            "warmups": 2, "repeats": 7, "operations_per_sample": 3, "dtype": "float16",
            "method": "original eight-key Python sparse dictionary versus per-lane rounded arithmetic",
            "excludes": "table/input construction,CUDA/libdevice,full65536table,shared copies,launches,graphs,HBM,model throughput",
            "rows": rows}


def build_report(latency):
    matrix = []
    for dtype in DTYPES:
        table = sparse_table(GATES[dtype], dtype)
        for g, u in pairs(dtype):
            r = lane(g, u, dtype)
            r.update(dtype=dtype, lookup_output_word=lookup(g, u, dtype, table))
            matrix.append(r)
    variants = [("cold_supported", {}), ("cold_capture", {"capturing": True}),
                ("warm_capture", {"capturing": True, "cached_keys": [["float16", 0]]}),
                ("wrong_dtype_cache", {"capturing": True, "cached_keys": [["bfloat16", 0]]}),
                ("wrong_device_cache", {"capturing": True, "cached_keys": [["float16", 1]]}),
                ("empty_warmup", {"tokens": 0}), ("rocm", {"hip": True}),
                ("sm9", {"major": 9}), ("small_shared", {"shared_optin_bytes": TABLE_BYTES - 1}),
                ("clamped", {"clamp": True}), ("other_activation", {"operation": "gelu"}),
                ("float32", {"dtype": "float32"})]
    float_key_table = {decode(g, "float16"): a for g, a in sparse_table([0, 0x8000], "float16").items()}
    zero_misses = sum(float_key_table[decode(g, "float16")] != g for g in (0, 0x8000))
    x = -89.0
    stable = x * math.exp(x) / (1 + math.exp(x))
    return {"schema_version": 1, "round": 158, "provenance": "synthetic", "independently_reproduced": False,
            "source": {"repository": "sgl-project/sglang", "pin": PIN,
                       "chronology": "actualNov14UTC06:20:47/BJT14:20:47 rounded SiLU LUT addition",
                       "license": "Apache-2.0 rights retained; original arithmetic/cache model, no implementation mirror",
                       "verified": "16-bit raw-index LUT rounds activation before FP32multiply;exact dtype/device cache;SMmajor10,128KiBoptin;warmcache precedes capture test",
                       "scope": "Python libm-exp rounded to FP32 is not __nv_expf;finite sparse fixture is not native full-domain byte parity"},
            "case": matrix[2], "matrix": matrix,
            "summary": {"lanes": len(matrix),
                        "single_round_misses": sum(r["rounded_output_word"] != r["single_round_output_word"] for r in matrix),
                        "lookup_misses": sum(r["rounded_output_word"] != r["lookup_output_word"] for r in matrix),
                        "zero_key_misses": zero_misses, "zero_key_cases": 2,
                        "native_table_entries_from_contract": 65536, "native_logical_table_bytes_from_contract": TABLE_BYTES,
                        "original_sparse_entries_per_dtype": len(GATES["float16"])},
            "dispatch_matrix": [{"name": name, "inputs": {**DEFAULT, **change},
                                 "outputs": route({**DEFAULT, **change})} for name, change in variants],
            "overflow_witness": {"dtype": "bfloat16", "gate_word": encode(x, "bfloat16"),
                                 "original_fp32_divide_activation_word": encode(activation(x), "bfloat16"),
                                 "original_high_precision_stable_activation_word": encode(stable, "bfloat16"),
                                 "native_libdevice_output_word": None},
            "latency": latency, "native_full65536_gate_word_parity_verified": None,
            "native_all_gate_up_pairs_parity_verified": None, "native_gpu_latency_ms": None,
            "native_allocator_peak_bytes": None, "browser_visual_acceptance": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.measure:
        (HERE / "latency.json").write_text(json.dumps(measure(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = build_report(json.loads((HERE / "latency.json").read_text()))
    from .render import outputs
    for file, text in {HERE / "report.json": json.dumps(report, indent=2, allow_nan=False) + "\n", **outputs(report)}.items():
        if args.check:
            if file.read_text(encoding="utf-8") != text:
                raise SystemExit(f"stale artifact: {file}")
        else:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
    print("Round158 sealed artifacts match" if args.check else "Wrote Round158 original artifacts")


if __name__ == "__main__":
    main()
