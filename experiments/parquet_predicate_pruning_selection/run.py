"""Original integer-column SQL truth-set pruning study; no DuckDB/Parquet execution."""

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import platform
import statistics
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "77e8a4971900bfea7161bd48e027fea589cb041e"
MAX_EXACT = 2**53-1
DEFAULT = {"row_groups":1024,"selected_groups":64,"columns":64,"projected_columns":8,
           "column_chunk_bytes":131072,"metadata_bytes":1048576}


@dataclass(frozen=True)
class Predicate:
    op: str
    column: str | None = None
    value: int | None = None
    children: tuple["Predicate", ...] = ()

    def __post_init__(self):
        if not isinstance(self.children,tuple) or not all(isinstance(p,Predicate) for p in self.children):
            raise ValueError("predicate children must be an immutable tuple")
        if self.op in ("eq","gt","is_null"):
            if not isinstance(self.column,str) or not self.column or self.children:
                raise ValueError("leaf requires a column and no children")
            if self.op=="is_null":
                if self.value is not None:
                    raise ValueError("IS NULL has no comparison literal")
            elif type(self.value) is not int:
                raise ValueError("integer comparison literal required; use IS NULL explicitly")
        elif self.op in ("and","or","not"):
            if self.column is not None or self.value is not None or len(self.children)!=(1 if self.op=="not" else 2):
                raise ValueError("logical predicate arity or fields invalid")
        else:
            raise ValueError("unsupported predicate")


@dataclass(frozen=True)
class ColumnStats:
    rows: int
    minimum: int | None
    maximum: int | None
    null_count: int | None

    def __post_init__(self):
        if type(self.rows) is not int or self.rows<0:
            raise ValueError("nonnegative row count required")
        if self.null_count is not None and (type(self.null_count) is not int or not 0<=self.null_count<=self.rows):
            raise ValueError("invalid null count")
        if (self.minimum is None)!=(self.maximum is None):
            raise ValueError("this bounded model requires both bounds or neither")
        if self.minimum is not None:
            if type(self.minimum) is not int or type(self.maximum) is not int or self.minimum>self.maximum:
                raise ValueError("ordered integer bounds required")
            if self.rows==0 or self.null_count==self.rows:
                raise ValueError("empty/all-null column cannot have non-null bounds")


def summarize(values):
    values = tuple(values)
    if any(v is not None and type(v) is not int for v in values):
        raise ValueError("only integer/null column values supported")
    valid = [v for v in values if v is not None]
    return ColumnStats(len(values),min(valid) if valid else None,max(valid) if valid else None,
                       len(values)-len(valid))


def sql_not(value):
    return None if value is None else not value


def sql_and(left,right):
    return False if left is False or right is False else None if left is None or right is None else True


def sql_or(left,right):
    return True if left is True or right is True else None if left is None or right is None else False


def evaluate(predicate,row):
    """Concrete row oracle; does not consult statistics."""
    op = predicate.op
    if op in ("and","or"):
        left,right = (evaluate(child,row) for child in predicate.children)
        return sql_and(left,right) if op=="and" else sql_or(left,right)
    if op=="not":
        return sql_not(evaluate(predicate.children[0],row))
    if predicate.column not in row:
        raise ValueError("row lacks a predicate column")
    value = row[predicate.column]
    if value is not None and type(value) is not int:
        raise ValueError("only integer/null column values supported")
    if op=="is_null":
        return value is None
    if value is None:
        return None
    return value==predicate.value if op=="eq" else value>predicate.value


def possible_truths(predicate,metadata):
    """Conservative product domain; deliberately forgets inter-predicate correlation."""
    op = predicate.op
    if op=="not":
        return {sql_not(value) for value in possible_truths(predicate.children[0],metadata)}
    if op in ("and","or"):
        left,right = (possible_truths(child,metadata) for child in predicate.children)
        combine = sql_and if op=="and" else sql_or
        return {combine(a,b) for a,b in itertools.product(left,right)}
    if predicate.column not in metadata:
        raise ValueError("metadata entry required; represent missing bounds explicitly")
    stats = metadata[predicate.column]
    null_possible = stats.rows>0 and (stats.null_count is None or stats.null_count>0)
    valid_possible = stats.rows>0 and (stats.null_count is None or stats.null_count<stats.rows)
    if op=="is_null":
        return ({True} if null_possible else set()) | ({False} if valid_possible else set())
    values = {None} if null_possible else set()
    if not valid_possible:
        return values
    if stats.minimum is None:
        return values | {True,False}
    if op=="eq":
        if stats.minimum<=predicate.value<=stats.maximum:
            values.add(True)
        if stats.minimum!=predicate.value or stats.maximum!=predicate.value:
            values.add(False)
    else:
        if stats.maximum>predicate.value:
            values.add(True)
        if stats.minimum<=predicate.value:
            values.add(False)
    return values


def metadata_for(rows):
    columns = set().union(*(row.keys() for row in rows))
    if not columns:
        columns = {"x"}
    if any(set(row)!=columns for row in rows):
        raise ValueError("row group requires a consistent column set")
    return {column:summarize(row[column] for row in rows) for column in columns}


def count_full(groups,predicate):
    return sum(evaluate(predicate,row) is True for rows in groups for row in rows)


def count_pruned(groups,metadata,predicate):
    if len(groups)!=len(metadata):
        raise ValueError("row-group metadata length mismatch")
    return sum(evaluate(predicate,row) is True for rows,stats in zip(groups,metadata)
               if True in possible_truths(predicate,stats) for row in rows)


def truth_names(values):
    return [label for value,label in ((True,"TRUE"),(False,"FALSE"),(None,"UNKNOWN")) if value in values]


def estimate_budget(p):
    for key in DEFAULT:
        value = p[key]
        minimum = 0 if key in ("selected_groups","metadata_bytes") else 1
        if type(value) is not int or not minimum<=value<=MAX_EXACT:
            raise ValueError(f"{key} must be a bounded exact integer >= {minimum}")
    if p["selected_groups"]>p["row_groups"] or p["projected_columns"]>p["columns"]:
        raise ValueError("selected groups/projected columns exceed declared totals")
    full = p["row_groups"]*p["columns"]*p["column_chunk_bytes"]
    projected = p["row_groups"]*p["projected_columns"]*p["column_chunk_bytes"]
    pruned = p["selected_groups"]*p["projected_columns"]*p["column_chunk_bytes"]
    result = {"unprojected_column_chunk_bytes":full,"projected_column_chunk_bytes":projected,
              "selected_column_chunk_bytes":pruned,"avoided_projected_chunk_bytes":projected-pruned,
              "unprojected_plus_metadata_bytes":full+p["metadata_bytes"],
              "projected_plus_metadata_bytes":projected+p["metadata_bytes"],
              "pruned_plus_metadata_bytes":pruned+p["metadata_bytes"],
              "metadata_bytes":p["metadata_bytes"]}
    if any(value>MAX_EXACT for value in result.values()):
        raise ValueError("budget exceeds exact integer range")
    return {**result,"actual_reader_bytes":None,"network_bytes":None,"native_scan_latency_ms":None,
            "peak_memory_bytes":None}


def fixture():
    return [[{"x":v} for v in values] for values in
            ((1,2,None,3),(10,None,11,12),(None,None,None,None),(5,6,7,10))]


def query():
    return Predicate("not",children=(Predicate("gt","x",9),))


def measure_latency():
    rows = []
    predicate = query()
    for group_count in (16,64,256,1024):
        groups = [[{"x":(5,6,7,10)[j%4] if i%8==0 else (10,11,12,None)[j%4]}
                   for j in range(256)] for i in range(group_count)]
        metadata = [metadata_for(group) for group in groups]
        expected = count_full(groups,predicate)
        assert count_pruned(groups,metadata,predicate)==expected
        primary = lambda:count_pruned(groups,metadata,predicate)
        baseline = lambda:count_full(groups,predicate)
        for _ in range(2):
            primary()
            baseline()
        samples = {"primary":[],"baseline":[]}
        for repeat in range(7):
            order = (("primary",primary),("baseline",baseline))
            for name,operation in order if repeat%2==0 else reversed(order):
                start = time.perf_counter_ns()
                for _ in range(3):
                    operation()
                samples[name].append((time.perf_counter_ns()-start)/3e6)
        rows.append({"row_groups":group_count,"rows":group_count*256,"selected_groups":group_count//8,
                     "matching_rows":expected,"result_count_error":0,
                     "primary_samples_ms":samples["primary"],"baseline_samples_ms":samples["baseline"],
                     "primary_median_ms":statistics.median(samples["primary"]),
                     "baseline_median_ms":statistics.median(samples["baseline"])})
    record = {"provenance":"measured","independently_reproduced":True,
              "implementation":"Original Python metadata gate plus retained-row predicate scan versus full predicate scan",
              "hardware":platform.processor() or platform.machine(),"python":platform.python_version(),
              "warmups":2,"repeats":7,"operations_per_sample":3,"order":"alternating",
              "exclusions":["metadata construction","Parquet decoding","filesystem/object store reads",
                            "DuckDB optimizer/native scan","network/cache/storage latency"],
              "rows":rows}
    (HERE/"latency.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")


def simulate():
    groups,predicate = fixture(),query()
    metadata = [metadata_for(rows) for rows in groups]
    kept = [i for i,stats in enumerate(metadata) if True in possible_truths(predicate,stats)]
    inner = predicate.children[0]
    wrong = [i for i,stats in enumerate(metadata) if not (True in possible_truths(inner,stats))]
    hidden = [{column:ColumnStats(s.rows,None,None,None) for column,s in stats.items()} for stats in metadata]
    missing = [i for i,stats in enumerate(hidden) if True in possible_truths(predicate,stats)]
    reversed_groups = [list(reversed(rows)) for rows in groups]
    reversed_kept = [i for i,rows in enumerate(reversed_groups)
                     if True in possible_truths(predicate,metadata_for(rows))]
    bounds = {"x":ColumnStats(2,7,7,None)}
    null_query = Predicate("is_null","x")
    constant_rows = [{"x":7},{"x":None}]
    wrong_constant_keep = True in possible_truths(null_query,{"x":ColumnStats(2,7,7,0)})
    lost_nulls = sum(evaluate(null_query,row) is True for row in constant_rows) if not wrong_constant_keep else 0
    correlation = [{"x":1,"y":0},{"x":0,"y":1}]
    conjunction = Predicate("and",children=(Predicate("eq","x",1),Predicate("eq","y",1)))
    changes = []
    for name,indices in (("Complete conservative metadata",kept),("Input row order reversed",reversed_kept),
                         ("All bounds/null counts missing",missing),("Wrong NOT of may-match",wrong)):
        matches = sum(evaluate(predicate,row) is True for i in indices for row in groups[i])
        reference = count_full(groups,predicate)
        changes.append({"name":name,"selected_groups":indices,"retained_matches":matches,
                        "change_matches":matches-reference,"retained_match_percent":100*matches/reference,
                        "change_retention_points":100*(matches-reference)/reference})
    return {"round":112,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"duckdb/duckdb","pin":PIN,"date":"2026-11-11T08:58:44Z","license":"MIT",
                      "paths":["extension/parquet/parquet_reader.cpp","extension/parquet/parquet_statistics.cpp",
                               "src/optimizer/statistics/expression/propagate_comparison.cpp"]},
            "case":{"query":"WHERE NOT (x > 9)","groups":[[row["x"] for row in rows] for rows in groups],
                    "group_truth_sets":[truth_names(possible_truths(predicate,stats)) for stats in metadata],
                    "selected_groups":kept,"total_rows":sum(map(len,groups)),
                    "candidate_rows":sum(len(groups[i]) for i in kept),
                    "exact_matching_rows":count_full(groups,predicate),
                    "pruned_matching_rows":count_pruned(groups,metadata,predicate),
                    "wrong_boolean_not_selected_groups":wrong,
                    "wrong_boolean_not_lost_matches":count_full(groups,predicate)-
                         sum(evaluate(predicate,row) is True for i in wrong for row in groups[i])},
            "constant_bounds":{"values":[7,None],"minimum":7,"maximum":7,"null_count":None,
                               "is_null_possible_truths":truth_names(possible_truths(null_query,bounds)),
                               "wrong_inferred_no_null_lost_matches":lost_nulls},
            "correlation":{"rows":correlation,"query":"x = 1 AND y = 1",
                           "metadata_may_match":True in possible_truths(conjunction,metadata_for(correlation)),
                           "actual_matching_rows":sum(evaluate(conjunction,row) is True for row in correlation)},
            "variants":changes,"budget":{"input":DEFAULT,**estimate_budget(DEFAULT)},
            "latency":json.loads((HERE/"latency.json").read_text(encoding="utf-8")),
            "unknowns":{"native_io_bytes":None,"native_latency_ms":None,"calibrated_confidence":None}}


def outputs():
    from experiments.parquet_predicate_pruning_selection.render import render_outputs
    report = simulate()
    return {HERE/"report.json":json.dumps(report,indent=2,ensure_ascii=False)+"\n",**render_outputs(report)}


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
            if not path.exists() or path.read_text(encoding="utf-8")!=content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 112 artifacts match." if args.check else "Wrote Round 112 artifacts.")


if __name__=="__main__":
    main()
