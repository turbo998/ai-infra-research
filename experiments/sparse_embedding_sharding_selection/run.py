"""Original jagged embedding pooling/sharding model and sealed CPU timing."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import time


HERE = Path(__file__).resolve().parent
TABLE = [[(r+1)*(c+1)/10 - ((r+c) % 3)/7 for c in range(6)] for r in range(8)]
BAGS = [[], [0], [0,0,1,7], [3,4], [6,7,6], [1,2,5,7]]


def validate(table: list, bags: list, mode: str, weights: list | None):
    if not table or not table[0] or any(len(row) != len(table[0]) for row in table):
        raise ValueError("table must be a nonempty rectangular matrix")
    if any(not math.isfinite(value) for row in table for value in row):
        raise ValueError("table values must be finite")
    if mode not in ("sum", "mean"):
        raise ValueError("pooling mode must be sum or mean")
    if any(type(index) is not int or not 0 <= index < len(table) for bag in bags for index in bag):
        raise ValueError("embedding id is outside real table rows")
    if weights is not None:
        if mode != "sum":
            raise ValueError("this harness validates weighted SUM only, not weighted MEAN")
        if len(weights) != len(bags) or any(len(w) != len(b) for w,b in zip(weights,bags)):
            raise ValueError("weights must match every bag occurrence")
        if any(not math.isfinite(w) for bag in weights for w in bag):
            raise ValueError("weights must be finite")


def dense_pool(table: list, bags: list, mode: str = "sum", weights: list | None = None) -> list:
    validate(table, bags, mode, weights)
    output = []
    for i, bag in enumerate(bags):
        denominator = len(bag) if mode == "mean" and bag else 1
        output.append([
            sum(table[index][column] * (weights[i][j] if weights is not None else 1)
                for j,index in enumerate(bag)) / denominator
            for column in range(len(table[0]))
        ])
    return output


def sharded_pool(
    table: list, bags: list, mode: str = "sum", weights: list | None = None,
    shards: int = 4, axis: str = "row", control: str = "exact",
) -> list:
    validate(table, bags, mode, weights)
    if type(shards) is not int or shards < 1:
        raise ValueError("shards must be a positive integer")
    if axis not in ("row", "column", "table") or control not in ("exact", "local-means", "dedup", "reverse-columns", "epsilon-mean"):
        raise ValueError("unknown sharding axis or control")
    rows, columns = len(table), len(table[0])
    output = [[0.0]*columns for _ in bags]
    for i, original in enumerate(bags):
        positions = list(range(len(original)))
        if control == "dedup":
            seen = set()
            positions = []
            for j,index in enumerate(original):
                if index not in seen:
                    positions.append(j)
                    seen.add(index)
        if axis == "row":
            width = math.ceil(rows/shards)
            for rank in range(shards):
                owned = [j for j in positions if original[j]//width == rank]
                denominator = len(owned) if control == "local-means" and mode == "mean" and owned else 1
                for column in range(columns):
                    output[i][column] += sum(
                        table[original[j]][column] * (weights[i][j] if weights is not None else 1)
                        for j in owned
                    ) / denominator
        else:
            width = math.ceil(columns/shards) if axis == "column" else columns
            pieces = []
            for start in range(0, columns, width):
                pieces.append([
                    sum(table[original[j]][column] * (weights[i][j] if weights is not None else 1) for j in positions)
                    for column in range(start, min(start+width, columns))
                ])
            if control == "reverse-columns":
                pieces.reverse()
            output[i] = [value for piece in pieces for value in piece]
        if mode == "mean" and control != "local-means" and original:
            denominator = len(original) + (1e-6 if control == "epsilon-mean" else 0)
            output[i] = [value / denominator for value in output[i]]
    return output


def table_gradient(table: list, bags: list, upstream: list, mode: str, weights: list | None = None) -> list:
    validate(table, bags, mode, weights)
    if len(upstream) != len(bags) or any(len(row) != len(table[0]) for row in upstream):
        raise ValueError("upstream must match pooled output shape")
    gradient = [[0.0]*len(table[0]) for _ in table]
    for i,bag in enumerate(bags):
        divisor = len(bag) if mode == "mean" and bag else 1
        for j,index in enumerate(bag):
            coefficient = (weights[i][j] if weights is not None else 1) / divisor
            for column in range(len(table[0])):
                gradient[index][column] += upstream[i][column] * coefficient
    return gradient


def global_root(group: list[int], local_root: int) -> int:
    if not group or len(set(group)) != len(group) or any(type(rank) is not int or rank < 0 for rank in group):
        raise ValueError("group must contain distinct nonnegative global ranks")
    if type(local_root) is not int or not 0 <= local_root < len(group):
        raise ValueError("local root is outside the process group")
    return group[local_root]


@dataclass(frozen=True)
class Budget:
    vocabulary: int = 100000000
    dimension: int = 128
    shards: int = 8
    global_bags: int = 4096
    ids_per_bag: int = 20
    weight_bytes: int = 2
    id_bytes: int = 8
    output_bytes: int = 2


def estimate(b: Budget) -> dict:
    for key,value in asdict(b).items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if b.shards > min(b.vocabulary,b.dimension):
        raise ValueError("this equal-shard comparison requires at least one row and column per shard")
    if b.weight_bytes not in (2,4) or b.output_bytes not in (2,4) or b.id_bytes not in (4,8):
        raise ValueError("unsupported scalar width")
    if b.vocabulary > 2**(8*b.id_bytes-1):
        raise ValueError("table rows exceed the signed ID range")
    full = b.vocabulary*b.dimension*b.weight_bytes
    ids = b.global_bags*b.ids_per_bag*b.id_bytes
    pooled = b.global_bags*b.dimension*b.output_bytes
    return {
        "input": asdict(b), "table_wise_peak_weight_bytes": full,
        "row_wise_peak_weight_bytes": math.ceil(b.vocabulary/b.shards)*b.dimension*b.weight_bytes,
        "column_wise_peak_weight_bytes": b.vocabulary*math.ceil(b.dimension/b.shards)*b.weight_bytes,
        "row_wise_logical_id_delivery_bytes": ids,
        "column_wise_logical_id_delivery_bytes": ids*b.shards,
        "row_wise_dense_partial_bytes_all_ranks": pooled*b.shards,
        "column_wise_output_piece_bytes_all_ranks": pooled,
        "gpu_latency_ms": None, "recommendation_quality": None,
        "scope": "one table; logical deliveries/partial materialization, not actual network wire bytes; no optimizer or activation memory",
    }


def compare(name: str, output: list, reference: list, interpretation: str = "reference-gate") -> dict:
    if len(output) != len(reference) or any(len(x) != len(y) for x,y in zip(output,reference)):
        raise ValueError("comparison requires equal pooled-vector shapes")
    error = max(abs(a-b) for x,y in zip(output,reference) for a,b in zip(x,y))
    return {"id":name,"maximum_absolute_error":error,"matches_ideal":error<1e-12,
            "interpretation":interpretation,"output":output,"confidence":None}


def measure_latency():
    rows = []
    table = [[math.sin(r+c)/3 for c in range(8)] for r in range(64)]
    for count in (16,64,256,1024):
        bags = [[(i*7+j*j)%64 for j in range(i%9)] for i in range(count)]
        paths = {
            "dense": lambda: dense_pool(table,bags,"mean"),
            "row_sharded": lambda: sharded_pool(table,bags,"mean",shards=4),
        }
        for call in paths.values():
            for _ in range(2):
                call()
        samples = {key:[] for key in paths}
        for repeat in range(7):
            for key in (list(paths) if repeat%2 == 0 else list(reversed(paths))):
                start = time.perf_counter_ns()
                paths[key]()
                samples[key].append(time.perf_counter_ns()-start)
        rows.append({"bags":count,"samples_ns":samples,
                     **{key+"_median_ms":statistics.median(value)/1e6 for key,value in samples.items()}})
    data = {
        "provenance":"measured","scope":"local CPU Python scalar functions including validation; no parallel workers, TorchRec, FBGEMM or GPU",
        "sealed_at_utc":datetime.now(timezone.utc).isoformat(),
        "environment":{"os":platform.system(),"architecture":platform.machine(),
                       "python":platform.python_version(),"processor":platform.processor() or None},
        "fixture":{"table_rows":64,"dimension":8,"row_shards":4,"bag_length":"i mod 9",
                   "id_formula":"(i*7+j*j) mod 64","pooling":"ideal unweighted mean"},
        "warmups_per_path":2,"repeats_per_path":7,"rows":rows,
    }
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8",newline="\n")


def load_latency() -> dict:
    path = HERE/"latency.json"
    if not path.exists():
        raise ValueError("missing sealed CPU timings; run with --measure before rendering")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["provenance"] != "measured" or data["repeats_per_path"] != 7:
        raise ValueError("invalid sealed timing provenance/repeat contract")
    if [row["bags"] for row in data["rows"]] != [16,64,256,1024]:
        raise ValueError("invalid timing load tiers")
    for row in data["rows"]:
        for key in ("dense","row_sharded"):
            samples = row["samples_ns"][key]
            if len(samples) != 7 or any(type(n) is not int or n <= 0 for n in samples):
                raise ValueError("sealed samples must be positive nanosecond measurements")
            if row[key+"_median_ms"] != statistics.median(samples)/1e6:
                raise ValueError("sealed median does not match samples")
    return data


def simulate() -> dict:
    reference = dense_pool(TABLE,BAGS,"mean")
    cases = [compare(axis+"-ideal-mean",sharded_pool(TABLE,BAGS,"mean",axis=axis),reference) for axis in ("table","row","column")]
    cases.extend([
        compare("sum-of-local-means",sharded_pool(TABLE,BAGS,"mean",control="local-means"),reference),
        compare("dedup-without-multiplicity",sharded_pool(TABLE,BAGS,"mean",control="dedup"),reference),
        compare("reversed-column-pieces",sharded_pool(TABLE,BAGS,"mean",axis="column",control="reverse-columns"),reference),
        compare("global-count-plus-epsilon",sharded_pool(TABLE,BAGS,"mean",control="epsilon-mean"),reference,"explicit numerical-policy difference, not an upstream bug claim"),
    ])
    return {
        "schema_version":1,"round":96,"provenance":"synthetic","independently_reproduced":False,
        "physical_gpu_measurement":False,
        "source":"https://github.com/meta-pytorch/torchrec/blob/36276ee3d0c9d0e52a00cbf10129035f529314b2/torchrec/distributed/embeddingbag.py",
        "fixture":{"table":TABLE,"bags":BAGS,"shards":4,"ideal_mean_tolerance":1e-12},
        "cases":cases,
        "perturbations":[
            compare("reverse-occurrences",sharded_pool(TABLE,[list(reversed(b)) for b in BAGS],"mean"),reference),
            compare("two-row-shards",sharded_pool(TABLE,BAGS,"mean",shards=2),reference),
        ],
        "subgroup":{"global_members":[1,3,5,7],"local_root":0,"mapped_global_root":global_root([1,3,5,7],0),
                    "literal_global_zero_is_member":False},
        "budget":estimate(Budget()),"latency":load_latency(),"quality":None,
        "notes":[
            "Exact ideal SUM and unweighted MEAN; weighted SUM separately tested. No weighted-MEAN API claim.",
            "Pinned RW/TWRW callback adds 1e-6 to its global count/weight divisor; not identical to ideal mathematical mean.",
            "Training RW uses sparse AlltoAll then pooled ReduceScatter; inference output paths differ.",
            "CPU timings measure serial Python functions, not distributed sharding acceleration.",
            "Budget logical ID deliveries include local delivery and do not count offsets, lengths or collective routing.",
        ],
    }


def outputs() -> dict[Path,str]:
    from .render import render_outputs
    report=simulate()
    return {HERE/"report.json":json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--measure",action="store_true",help="explicitly replace the sealed local CPU timing samples")
    args=parser.parse_args()
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
    print("Round 96 artifacts match." if args.check else "Wrote Round 96 artifacts.")


if __name__ == "__main__":
    main()
