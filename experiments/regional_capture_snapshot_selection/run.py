"""Original bounded one-dimensional storage/capture model, not PyTorch execution."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "7bbc31dd45f4e43a5e3899e7e328aedfd967f2ff"
DEFAULT = {"offset": 3, "length": 4, "stride": 2, "element_bytes": 4}


def layout_budget(p):
    if set(p) != set(DEFAULT) or any(type(value) is not int or value < 0 for value in p.values()):
        raise ValueError("exact nonnegative layout fields required")
    if p["element_bytes"] not in (1, 2, 4, 8) or p["length"] < 1:
        raise ValueError("nonempty original 1-D view and 1/2/4/8 byte width required")
    span = p["offset"] + 1 + (p["length"] - 1) * p["stride"]
    out = {"logical_bytes": p["length"] * p["element_bytes"],
           "layout_preserving_buffer_bytes": span * p["element_bytes"]}
    if max(*p.values(), *out.values()) > 2**53 - 1:
        raise ValueError("exact integer budget range exceeded")
    return out


@dataclass(eq=False)
class View:
    storage: list
    offset: int
    length: int
    stride: int

    def __post_init__(self):
        layout_budget({"offset": self.offset, "length": self.length,
                       "stride": self.stride, "element_bytes": 4})
        if self.offset + (self.length - 1) * self.stride >= len(self.storage):
            raise ValueError("view exceeds backing storage")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in self.storage):
            raise ValueError("finite scalar backing values required")

    def values(self):
        return [self.storage[self.offset + i * self.stride] for i in range(self.length)]


def snapshot(tree, clone=False, state_storages=()):
    """Supported inputs: acyclic lists/tuples/string-keyed dicts, View and primitives."""
    memo, seen_storages, active = {}, set(), set()

    def copy(value):
        if value is None or type(value) in (str, bool, int, float):
            return value
        key = id(value)
        if key in active:
            raise ValueError("cyclic containers outside original model scope")
        if key in memo:
            return memo[key]
        if isinstance(value, View):
            storage_id = id(value.storage)
            if clone and storage_id in state_storages:
                raise ValueError("clone would break alias with selected region state")
            if clone and storage_id in seen_storages:
                raise ValueError("distinct views of shared storage cannot preserve alias under this clone policy")
            seen_storages.add(storage_id)
            span = value.offset + 1 + (value.length - 1) * value.stride
            backing = list(value.storage[:span]) if clone else value.storage
            result = View(backing, value.offset, value.length, value.stride)
        elif type(value) in (list, tuple, dict):
            active.add(key)
            if isinstance(value, dict):
                if not all(type(k) is str for k in value):
                    raise ValueError("original dictionary keys must be strings")
                result = {k: copy(v) for k, v in value.items()}
            else:
                items = [copy(v) for v in value]
                result = tuple(items) if type(value) is tuple else items
            active.remove(key)
        else:
            raise ValueError("unsupported object: native arbitrary objects/subclasses not reproduced")
        memo[key] = result
        return result

    return copy(tree)


def capture_events(events, aliases, clone=False):
    """Original scalar module-boundary event model: capture precedes a +1 hook."""
    calls = {module: [] for module in set(aliases.values())}
    for path, view in events:
        if path not in aliases:
            raise ValueError("unknown module path")
        calls[aliases[path]].append(snapshot((view,), clone))
        for index in {view.offset + i * view.stride for i in range(view.length)}:
            view.storage[index] += 1
    return {path: {"status": None if calls[module] else "not_reached",
                   "calls": [call[0].values() for call in calls[module]]}
            for path, module in aliases.items()}


def measure():
    rows = []
    for n in (64, 256, 1024, 4096):
        view = View(list(range(3 + 2 * n)), 3, n, 2)
        functions = (lambda: snapshot([view], False), lambda: snapshot([view], True))
        assert functions[0]()[0].values() == functions[1]()[0].values()
        for _ in range(2):
            for function in functions:
                function()
        samples = [[], []]
        for repeat in range(7):
            for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                started = time.perf_counter_ns()
                for _ in range(5):
                    functions[i]()
                samples[i].append((time.perf_counter_ns() - started) / 5 / 1e6)
        rows.append({"logical_elements": n, "detach_samples_ms": samples[0], "clone_samples_ms": samples[1],
                     "detach_median_ms": statistics.median(samples[0]), "clone_median_ms": statistics.median(samples[1])})
    return {"provenance": "measured", "python": platform.python_version(), "machine": platform.machine(),
            "warmups": 2, "repeats": 7, "operations_per_sample": 5,
            "method": "original Python view snapshot including finite backing validation, detach versus prefix-layout clone; different ownership semantics",
            "excludes": "PyTorch tensors, native hooks, compilation, GPU, autograd and arbitrary object traversal", "rows": rows}


def build_report(latency):
    backing = list(range(12))
    view = View(backing, 3, 4, 2)
    detached = snapshot({"x": view, "same": view})
    cloned = snapshot({"x": view, "same": view}, True)
    before = view.values()
    backing[5] = 50
    after = view.values()
    aliases = {"left": "shared", "right": "shared", "unused": "never"}
    def event_case(clone):
        v = View([1, 2, 3], 0, 3, 1)
        return capture_events([("left", v), ("right", v)], aliases, clone)
    return {"schema_version": 1, "round": 124, "provenance": "synthetic", "independently_reproduced": False,
            "source": {"repository": "pytorch/pytorch", "pin": PIN, "date_utc": "2026-11-14T00:44:56Z",
                       "license": "BSD-style", "change": "regional AOTI module-boundary capture; contract validation explicitly subsequent work"},
            "case": {"before": before, "live_after_mutation": after, "detached_after": detached["x"].values(),
                     "cloned_after": cloned["x"].values(), "detach_preserved_coordinates": sum(a == b for a,b in zip(before, detached["x"].values())),
                     "clone_preserved_coordinates": sum(a == b for a,b in zip(before, cloned["x"].values())),
                     "coordinates": len(before), "same_leaf_identity_preserved": cloned["x"] is cloned["same"],
                     "layout": DEFAULT, "budget": layout_budget(DEFAULT),
                     "live_event_capture": event_case(False), "cloned_event_capture": event_case(True)},
            "latency": latency, "native_aoti_latency_ms": None, "native_compilation_reproduced": False,
            "native_peak_hbm_bytes": None, "native_arbitrary_object_capture_reproduced": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(), indent=2)+"\n", encoding="utf-8")
    report = build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for path, text in {HERE/"report.json": json.dumps(report, indent=2)+"\n", **outputs(report)}.items():
        if args.check:
            if path.read_text(encoding="utf-8") != text:
                raise SystemExit(f"stale artifact: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    print("Round124 sealed artifacts match" if args.check else "Wrote Round124 original artifacts")


if __name__ == "__main__":
    main()
