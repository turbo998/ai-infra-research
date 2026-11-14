"""Original page-owner ledger, not a storage plugin, allocator or copy kernel."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

HERE = Path(__file__).resolve().parent
PIN = "56fa7ca6e7e4004aeb935b6bb38d0acf81340510"
DEFAULT = {
    "page_size": 16, "shards": 4, "rank": 1,
    "backend": "rank-local-demo", "registered": ["rank-local-demo"], "hicache": True,
    "prefix_tokens": 32, "found_at_alloc": 32, "base_at_alloc": 1,
    "found_at_admit": 32, "base_at_admit": 1, "joint_tokens": 0,
    "free_pages": [8, 6, 7, 5], "fetch_pages": 7,
}


def integer(v, maximum=2**20):
    return type(v) is int and 0 <= v <= maximum


def validate(p):
    if not isinstance(p, dict) or set(p) != set(DEFAULT):
        raise ValueError("exact original page-owner fields required")
    if not integer(p["page_size"]) or not p["page_size"] or not integer(p["shards"], 16) or not p["shards"]:
        raise ValueError("positive bounded physical page size and shard count required")
    n = p["shards"]
    if not integer(p["rank"], n-1) or type(p["hicache"]) is not bool:
        raise ValueError("rank in shard group and explicit HiCache witness required")
    if not isinstance(p["backend"], str) or not p["backend"]:
        raise ValueError("named original backend required")
    reg = p["registered"]
    if not isinstance(reg, list) or len(reg) > 16 or any(not isinstance(v, str) or not v for v in reg):
        raise ValueError("bounded named registration witnesses required")
    if len(set(reg)) != len(reg):
        raise ValueError("duplicate registry witnesses are ambiguous")
    for k in ("prefix_tokens", "found_at_alloc", "found_at_admit", "joint_tokens"):
        if not integer(p[k]) or p[k] % p["page_size"]:
            raise ValueError("original profile uses aligned token-prefix observations")
    for k in ("base_at_alloc", "base_at_admit"):
        if p[k] is not None and not integer(p[k], n-1):
            raise ValueError("observed rotation base must be null or an owner class")
    if not integer(p["fetch_pages"], 128):
        raise ValueError("bounded original whole-page fetch required")
    total = p["prefix_tokens"] + p["fetch_pages"] * p["page_size"]
    if p["found_at_alloc"] > p["prefix_tokens"] or p["found_at_admit"] > total or p["joint_tokens"] > total:
        raise ValueError("observed prefix exceeds declared key span")
    if not isinstance(p["free_pages"], list) or len(p["free_pages"]) != n or any(not integer(v) for v in p["free_pages"]):
        raise ValueError("one bounded observed free-page count per owner class required")


def counts(start, pages, shards):
    return [pages // shards + int((c-start) % shards < pages % shards) for c in range(shards)]


def buckets(pages, shards):
    out = [[] for _ in range(shards)]
    for h in pages:
        out[h % shards].append(h // shards)
    return out


def plan(p, detached_proposal=False):
    validate(p)
    n = p["shards"]
    supported = p["backend"] in p["registered"]
    startup = not (n > 1 and p["hicache"] and not supported)
    attach = p["hicache"] and (n == 1 or supported)
    result = {
        "startup_admitted": startup, "runtime_attachment_admitted": attach,
        "actual_legacy_startup_admitted": not (n > 1 and p["hicache"]),
        "action": "not_initialized" if not p["hicache"] else "unregistered_backend",
        "raw_start_owner": None, "owners": [], "host_logical_pages": [],
        "owned_local_host_pages": [], "page_draw_counts": [],
        "recovered_rotation_base": None, "published_rotation_base": None,
        "owner_misses": None, "owned_token_rows": None,
        "native_plugin_verified": None, "native_copy_verified": None,
    }
    if not startup or not attach:
        return result
    least = max(range(n), key=lambda c: (p["free_pages"][c], -c))
    base = p["base_at_alloc"] if p["found_at_alloc"] else None
    if base is None:
        base = least
    offset = p["prefix_tokens"] // p["page_size"]
    desired = base + offset
    start = least if detached_proposal else desired
    draws = counts(start, p["fetch_pages"], n)
    result.update(raw_start_owner=start, page_draw_counts=draws)
    if any(want > free for want, free in zip(draws, p["free_pages"])):
        result["action"] = "attempt_capacity_miss"
        return result
    owners = [(start+j) % n for j in range(p["fetch_pages"])]
    used = [0] * n
    host = []
    for c in owners:
        used[c] += 1
        host.append(used[c] * n + c)
    own = buckets(host, n)[p["rank"]]
    recovered = (owners[0]-offset) % n if owners else None
    misses = sum(c != (desired+j) % n for j, c in enumerate(owners))
    result.update(owners=owners, host_logical_pages=host, owned_local_host_pages=own,
                  recovered_rotation_base=recovered, owner_misses=misses,
                  owned_token_rows=len(own)*p["page_size"])
    total = p["prefix_tokens"] + p["fetch_pages"]*p["page_size"]
    if p["joint_tokens"] >= total:
        result["action"] = "device_covered"
    elif not owners:
        result["action"] = "no_full_span"
    elif not detached_proposal and p["found_at_admit"] > 0 and p["base_at_admit"] != recovered:
        result["action"] = "refetch_rotation"
    else:
        result["action"] = "publish_rotation"
        result["published_rotation_base"] = 0 if detached_proposal else recovered
    return result


def measure():
    rows = []
    for size in (64, 256, 1024, 4096):
        pages = list(range(7, 7+size))
        def single():
            return buckets(pages, 4)
        def scans():
            return [[h//4 for h in pages if h % 4 == c] for c in range(4)]
        functions = (single, scans)
        if single() != scans():
            raise RuntimeError("original owner partitions differ")
        for _ in range(2):
            for f in functions:
                f()
        samples = [[], []]
        for repeat in range(7):
            for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start = time.perf_counter_ns()
                for _ in range(3):
                    functions[i]()
                samples[i].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"pages": size, "single_samples_ms": samples[0], "scan_samples_ms": samples[1],
                     "single_median_ms": statistics.median(samples[0]), "scan_median_ms": statistics.median(samples[1])})
    return {"provenance": "measured", "python": platform.python_version(), "machine": platform.machine(),
            "warmups": 2, "repeats": 7, "operations_per_sample": 3,
            "method": "original CPU metadata partition,one pass versus four independent scans",
            "excludes": "setup,Torch allocators,native storage/copy/IO,network,HBM and plugin throughput", "rows": rows}


def build_report(latency):
    variants = [
        ("registered_chain", DEFAULT),
        ("unregistered", {**DEFAULT, "registered": []}),
        ("rotation_changed", {**DEFAULT, "base_at_admit": 2}),
        ("prefix_evicted", {**DEFAULT, "found_at_admit": 0, "base_at_admit": None}),
        ("new_chain", {**DEFAULT, "prefix_tokens": 0, "found_at_alloc": 0, "base_at_alloc": None,
                       "found_at_admit": 0, "base_at_admit": None}),
        ("capacity_miss", {**DEFAULT, "free_pages": [0, 0, 0, 0]}),
        ("joint_device_covered", {**DEFAULT, "joint_tokens": 144, "base_at_admit": 2}),
        ("no_full_span", {**DEFAULT, "fetch_pages": 0}),
    ]
    return {
        "schema_version": 1, "round": 162, "provenance": "synthetic", "independently_reproduced": False,
        "source": {"repository": "sgl-project/sglang", "pin": PIN,
                   "chronology": "actualNov14UTC02:09:23/BJT10:09:23 out-of-tree logical-page HiCache admission",
                   "license": "Apache-2.0 rights retained,no implementation mirror",
                   "verified": "empty upstream registry;startup and runtime guards;start owner continues prefix;least-full seeding;staged base recovery;rotation warning/refetch;insert preserves recovered base",
                   "scope": "original aligned whole-page metadata ledger,not a native plugin,allocator or storage/copy kernel"},
        "case": {"inputs": DEFAULT, "current": plan(DEFAULT), "detached_proposal": plan(DEFAULT, True),
                 "baseline_label": "counterfactual detached rotation proposal,not native old execution;actual old startup rejected sharded storage"},
        "matrix": [{"name": name, "inputs": p, "outputs": plan(p)} for name, p in variants],
        "latency": latency, "native_plugin_verified": None, "native_copy_verified": None,
        "browser_visual_acceptance": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(), indent=2)+"\n", encoding="utf-8")
    r = build_report(json.loads((HERE/"latency.json").read_text()))
    from .render import outputs
    for file, text in {HERE/"report.json": json.dumps(r, indent=2)+"\n", **outputs(r)}.items():
        if a.check:
            if file.read_text(encoding="utf-8") != text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
    print("Round162 sealed artifacts match" if a.check else "Wrote Round162 original artifacts")


if __name__ == "__main__":
    main()
