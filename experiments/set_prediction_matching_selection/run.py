"""Original set-assignment references and distributed object-loss arithmetic."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import time


HERE = Path(__file__).resolve().parent
SOURCE = "https://github.com/lyuwenyu/RT-DETR/blob/a4a0c6a3ea2ae7bd5d232c19486a7ebd9702f19c/rtdetr_pytorch/src/zoo/rtdetr/matcher.py"
COST = [[1.0,2.0],[2.0,100.0],[80.0,80.0]]
PREDICTIONS = [0.0,2.0,9.0]
TARGETS = [1.0,0.0]
MAX_SAFE = 9007199254740991


def shape(cost: list) -> tuple[int,int]:
    if not cost or any(len(row) != len(cost[0]) for row in cost):
        raise ValueError("cost matrix needs equally sized query rows")
    queries,targets = len(cost),len(cost[0])
    if targets > queries:
        raise ValueError("this toy requires enough queries to match every target")
    if any(not math.isfinite(value) for row in cost for value in row):
        raise ValueError("all costs must be finite")
    return queries,targets


def assignment_cost(cost: list,assignment: tuple) -> float:
    queries,targets = shape(cost)
    if len(assignment) != targets or len(set(assignment)) != targets or any(
            type(q) is not int or not 0 <= q < queries for q in assignment):
        raise ValueError("assignment must use one distinct valid query per target")
    return math.fsum(cost[q][t] for t,q in enumerate(assignment))


def enumerate_assignment(cost: list) -> dict:
    queries,targets = shape(cost)
    if queries > 9:
        raise ValueError("exhaustive reference is capped at nine queries")
    score,assignment = min((math.fsum(cost[q][t] for t,q in enumerate(order)),order)
                           for order in itertools.permutations(range(queries),targets))
    return {"cost":score,"query_for_target":assignment}


def subset_assignment(cost: list) -> dict:
    """Independent target-subset DP, advancing each query exactly once."""
    queries,targets = shape(cost)
    if targets > 12:
        raise ValueError("exponential reference is capped at twelve targets")
    best = {0:(0.0,(-1,)*targets)}
    for query in range(queries):
        following = dict(best)
        for mask,(score,assignment) in best.items():
            for target in range(targets):
                if mask & (1 << target):
                    continue
                order = assignment[:target]+(query,)+assignment[target+1:]
                candidate = (score+cost[query][target],order)
                key = mask | (1 << target)
                if key not in following or candidate < following[key]:
                    following[key] = candidate
        best = following
    score,assignment = best[(1 << targets)-1]
    return {"cost":score,"query_for_target":assignment}


def greedy_assignment(cost: list) -> dict:
    queries,targets = shape(cost)
    selected,used = {},set()
    for value,query,target in sorted((cost[q][t],q,t) for q in range(queries) for t in range(targets)):
        if query not in used and target not in selected:
            selected[target] = query
            used.add(query)
    order = tuple(selected[t] for t in range(targets))
    return {"cost":assignment_cost(cost,order),"query_for_target":order}


def scalar_loss(predictions: list,targets: list,assignment: tuple | None = None) -> dict:
    if not predictions or any(not math.isfinite(x) for x in predictions+targets):
        raise ValueError("finite scalar predictions and targets are required")
    cost = [[(prediction-target)**2 for target in targets] for prediction in predictions]
    if assignment is None:
        assignment = subset_assignment(cost)["query_for_target"]
    assignment_cost(cost,assignment)
    gradient = [0.0]*len(predictions)
    values = []
    for target,query in enumerate(assignment):
        residual = predictions[query]-targets[target]
        gradient[query] += residual
        values.append(0.5*residual**2)
    return {"loss":math.fsum(values),"gradient":gradient,"query_for_target":assignment}


def distributed_gradient(counts: list,sums: list,policy: str = "source_global") -> dict:
    if policy not in ("source_global","rank_local","global_without_world_compensation"):
        raise ValueError("unknown normalization policy")
    if not counts or len(counts) != len(sums) or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("one nonnegative integer count and gradient sum are required per rank")
    if any(not math.isfinite(value) or (n == 0 and value != 0) for n,value in zip(counts,sums)):
        raise ValueError("finite matched-coordinate gradient sums must be zero on empty ranks")
    world,total = len(counts),sum(counts)
    if policy == "source_global":
        divisors = [max(total/world,1.0)]*world
    elif policy == "rank_local":
        divisors = [max(n,1) for n in counts]
    else:
        divisors = [max(total,1)]*world
    return {"counts":counts,"gradient_sums":sums,"policy":policy,"per_rank_denominators":divisors,
            "ddp_average_gradient":math.fsum(value/divisor for value,divisor in zip(sums,divisors))/world,
            "source_equivalent_reference":math.fsum(sums)/max(total,world),
            "ideal_object_mean":math.fsum(sums)/max(total,1)}


@dataclass(frozen=True)
class Budget:
    batch: int = 8
    queries: int = 300
    targets_per_image: int = 40
    cost_bytes: int = 4
    auxiliary_layers: int = 5


def estimate(b: Budget) -> dict:
    for name in asdict(b):
        value = getattr(b,name)
        if type(value) is not int or not (0 if name in ("targets_per_image","auxiliary_layers") else 1) <= value <= MAX_SAFE:
            raise ValueError(f"{name} must be a nonnegative/positive safe integer")
    if b.targets_per_image > b.queries or b.cost_bytes not in (4,8):
        raise ValueError("enough queries and 4/8-byte costs required")
    block = b.batch*b.queries*b.targets_per_image
    flat = b.batch*block
    passes = 1+b.auxiliary_layers
    values = (block,flat,block*b.cost_bytes,flat*b.cost_bytes,flat*b.cost_bytes*passes)
    if passes > MAX_SAFE or any(value > MAX_SAFE for value in values):
        raise ValueError("cost-work counter exceeds exact numerical range")
    return {"input":asdict(b),"per_image_block_elements":block,"flattened_cross_image_elements":flat,
            "per_image_block_bytes":block*b.cost_bytes,"flattened_cross_image_bytes":flat*b.cost_bytes,
            "main_and_aux_matching_passes":passes,"flattened_cost_bytes_written_across_passes":flat*b.cost_bytes*passes,
            "peak_memory_bytes":None,"gpu_latency_ms":None,"detector_ap":None,
            "scope":"uniform target counts; raw cost arrays, not peak memory or execution time; excludes denoising"}


def timing_fixture(targets: int) -> list:
    return [[((q+3)*(t+5)%17)+0.07*q+0.013*t for t in range(targets)] for q in range(targets+2)]


def measure_latency():
    rows = []
    for targets in (2,3,4,5):
        cost = timing_fixture(targets)
        functions = {"subset_dp":lambda:subset_assignment(cost),
                     "enumeration":lambda:enumerate_assignment(cost)}
        if abs(functions["subset_dp"]()["cost"]-functions["enumeration"]()["cost"]) > 1e-12:
            raise ValueError("timed exact references disagree")
        for function in functions.values():
            for _ in range(2):
                function()
        samples = {key:[] for key in functions}
        for repeat in range(7):
            for key in list(functions) if repeat%2 == 0 else list(reversed(functions)):
                start = time.perf_counter_ns()
                functions[key]()
                samples[key].append(time.perf_counter_ns()-start)
        rows.append({"queries":targets+2,"targets":targets,"samples_ns":samples,
                     **{key+"_median_ms":statistics.median(values)/1e6 for key,values in samples.items()}})
    data = {"provenance":"measured","measured_at_utc":datetime.now(timezone.utc).isoformat(),
            "environment":{"os":platform.system(),"machine":platform.machine(),
                           "processor":platform.processor(),"python":platform.python_version()},
            "scope":"original exponential DP versus exhaustive reference including validation; not SciPy Hungarian or detector execution",
            "timer":"perf_counter_ns","warmups_per_path":2,"repeats_per_path":7,
            "order":"alternating function order by repeat","rows":rows}
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8",newline="\n")


def load_latency() -> dict:
    path = HERE/"latency.json"
    if not path.exists():
        raise ValueError("missing sealed timing file; explicitly run --measure")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["provenance"] != "measured" or data["repeats_per_path"] != 7:
        raise ValueError("invalid measured timing contract")
    if [(row["queries"],row["targets"]) for row in data["rows"]] != [(4,2),(5,3),(6,4),(7,5)]:
        raise ValueError("sealed timing tiers do not match")
    for row in data["rows"]:
        for key in ("subset_dp","enumeration"):
            samples = row["samples_ns"][key]
            if len(samples) != 7 or any(type(n) is not int or n <= 0 for n in samples):
                raise ValueError("timing samples must be seven positive nanosecond integers")
            if row[key+"_median_ms"] != statistics.median(samples)/1e6:
                raise ValueError("sealed median differs from samples")
    return data


def simulate() -> dict:
    exact,dp,greedy = enumerate_assignment(COST),subset_assignment(COST),greedy_assignment(COST)
    duplicate = tuple(min(range(len(COST)),key=lambda q:COST[q][t]) for t in range(2))
    baseline = scalar_loss(PREDICTIONS,TARGETS)
    reversed_targets = scalar_loss(PREDICTIONS,list(reversed(TARGETS)))
    order = [2,0,1]
    reordered = [PREDICTIONS[i] for i in order]
    fresh,stale = scalar_loss(reordered,TARGETS),scalar_loss(reordered,TARGETS,baseline["query_for_target"])

    def restore(values):
        result = [0.0]*len(order)
        for new,old in enumerate(order):
            result[old] = values[new]
        return result

    finite_differences = []
    for query in range(len(PREDICTIONS)):
        plus,minus = PREDICTIONS[:],PREDICTIONS[:]
        plus[query] += 1e-5
        minus[query] -= 1e-5
        value = (scalar_loss(plus,TARGETS)["loss"]-scalar_loss(minus,TARGETS)["loss"])/2e-5
        finite_differences.append(value)
    counts,sums = [1,3,0],[2.0,18.0,0.0]
    probability_cost = [[-0.60,-0.39],[-0.30,-0.15]]
    nll_cost = [[-math.log(-p) for p in row] for row in probability_cost]
    return {
        "schema_version":1,"round":102,"provenance":"synthetic","independently_reproduced":False,
        "physical_gpu_measurement":False,"source":SOURCE,"cost_matrix":COST,
        "assignment":{"exact":exact,"subset_dp":dp,"greedy":greedy,
                      "independent_target_argmin":{"query_for_target":duplicate,"valid_one_to_one":False,
                                                  "invalid_summed_cost":math.fsum(COST[q][t] for t,q in enumerate(duplicate))}},
        "scalar_supervision":{"predictions":PREDICTIONS,"targets":TARGETS,"baseline":baseline,
                              "reversed_targets":reversed_targets,"query_order":order,
                              "fresh_reordered":fresh,"stale_reordered":stale,
                              "fresh_gradient_restored":restore(fresh["gradient"]),
                              "stale_gradient_restored":restore(stale["gradient"]),
                              "finite_differences":finite_differences,
                              "finite_difference_error":max(abs(a-b) for a,b in zip(baseline["gradient"],finite_differences))},
        "cost_objective_switch":{"probability_cost":probability_cost,"negative_log_cost":nll_cost,
                                 "negative_probability_assignment":subset_assignment(probability_cost),
                                 "negative_log_assignment":subset_assignment(nll_cost),
                                 "intentional_objective_change":True},
        "normalization":[distributed_gradient(counts,sums,policy) for policy in
                         ("source_global","rank_local","global_without_world_compensation")],
        "low_count_clamp":distributed_gradient([1,0,0,0],[2.0,0,0,0]),
        "empty_coordinate_loss":distributed_gradient([0,0],[0.0,0.0]),
        "budget":estimate(Budget()),"latency":load_latency(),"detector_quality":None,
        "notes":[
            "Exact toy references are not Hungarian algorithm implementations or SciPy runtime benchmarks.",
            "Scalar squared loss is original, not box L1/GIoU/focal loss reproduction.",
            "Source matcher is no_grad; derivative checks stay within a unique-assignment region.",
            "Unmatched query coordinate gradient is zero; source classification still supervises no-object queries.",
            "DDP averages gradients; source count clamp means denominator max(global_objects, world_size), not always global_objects.",
            "Source cost tensor is cross-image-sized but only same-image blocks are matched; aux layers rematch.",
        ],
    }


def outputs() -> dict[Path,str]:
    from .render import render_outputs
    report = simulate()
    return {HERE/"report.json":json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--measure",action="store_true")
    args = parser.parse_args()
    if args.check and args.measure:
        parser.error("--check cannot replace sealed measurements")
    if args.measure:
        measure_latency()
    for path,content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 102 artifacts match." if args.check else "Wrote Round 102 artifacts.")


if __name__ == "__main__":
    main()
