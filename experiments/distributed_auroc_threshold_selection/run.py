"""Original binary ROC/pairwise oracle and mergeable threshold counters."""

import argparse
import itertools
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "e843b84bb9e957514deb8971340ec6c6e3f0887c"
MAX_SAFE = 2**53-1
DEFAULT = dict(rank_counts=[250000]*4,threshold_count=128,prediction_bytes=4,
               target_bytes=8,threshold_bytes=4,update_samples=50000)


class UndefinedMetric(ValueError):
    pass


def integer(value,minimum=0):
    return isinstance(value,int) and not isinstance(value,bool) and value>=minimum


def validate_pairs(pairs):
    for score,label in pairs:
        if (not isinstance(score,(int,float)) or isinstance(score,bool) or
                not math.isfinite(score) or not 0<=score<=1 or
                not integer(label) or label not in (0,1)):
            raise ValueError("finite probability scores in [0,1] and integer binary labels required")


def class_counts(pairs):
    validate_pairs(pairs)
    positives = sum(label for _,label in pairs)
    negatives = len(pairs)-positives
    if not positives or not negatives:
        raise UndefinedMetric("global positive and negative classes are both required")
    return positives,negatives


def integral(points):
    return float(sum((x1-x0)*(y0+y1)/2 for (x0,y0),(x1,y1) in zip(points,points[1:])))


def exact_roc(pairs):
    positives,negatives = class_counts(pairs)
    tp,fp = 0,0
    points = [(Fraction(0),Fraction(0))]
    for _,group in itertools.groupby(sorted(pairs,key=lambda row:row[0],reverse=True),key=lambda row:row[0]):
        labels = [label for _,label in group]
        tp += sum(labels)
        fp += len(labels)-sum(labels)
        points.append((Fraction(fp,negatives),Fraction(tp,positives)))
    return integral(points),points


def pairwise_auc(pairs):
    positives,negatives = class_counts(pairs)
    pos = [s for s,label in pairs if label==1]
    neg = [s for s,label in pairs if label==0]
    doubled_credit = sum(2*(p>n)+(p==n) for p in pos for n in neg)
    return doubled_credit/(2*positives*negatives)


@dataclass(frozen=True)
class ThresholdState:
    thresholds: tuple
    cells: tuple

    def __post_init__(self):
        if not isinstance(self.thresholds,tuple) or not isinstance(self.cells,tuple):
            raise ValueError("immutable threshold and cell tuples required")
        if (len(self.thresholds)<2 or self.thresholds[0]!=0 or self.thresholds[-1]!=1 or
                any(not isinstance(t,(int,float)) or isinstance(t,bool) or not math.isfinite(t) for t in self.thresholds) or
                any(a>=b for a,b in zip(self.thresholds,self.thresholds[1:]))):
            raise ValueError("strictly increasing finite threshold grid from 0 to 1 required")
        if (len(self.cells)!=len(self.thresholds) or
                any(not isinstance(row,tuple) or len(row)!=4 or any(not integer(n) for n in row) for row in self.cells)):
            raise ValueError("each threshold needs immutable TN,FP,FN,TP nonnegative integer cells")
        totals = {(tn+fp,fn+tp) for tn,fp,fn,tp in self.cells}
        if len(totals)!=1:
            raise ValueError("class counts must be consistent across thresholds")
        if self.cells[0][0] or self.cells[0][2]:
            raise ValueError("threshold zero must predict every bounded score positive")
        if any(a[1]<b[1] or a[3]<b[3] for a,b in zip(self.cells,self.cells[1:])):
            raise ValueError("positive predictions cannot increase as thresholds increase")


def threshold_counts(pairs,thresholds):
    validate_pairs(pairs)
    rows = []
    for threshold in thresholds:
        tn=fp=fn=tp=0
        for score,label in pairs:
            predicted = score>=threshold
            if label:
                tp += predicted
                fn += not predicted
            else:
                fp += predicted
                tn += not predicted
        rows.append((tn,fp,fn,tp))
    return ThresholdState(tuple(thresholds),tuple(rows))


def merge_states(states):
    if not states:
        raise ValueError("at least one declared threshold state required")
    thresholds = states[0].thresholds
    if any(state.thresholds!=thresholds for state in states):
        raise ValueError("cannot merge states built with different threshold grids")
    rows = tuple(tuple(sum(state.cells[i][j] for state in states) for j in range(4))
                 for i in range(len(thresholds)))
    return ThresholdState(thresholds,rows)


def threshold_roc(state,complete_origin=False):
    negatives,positives = state.cells[0][0]+state.cells[0][1],state.cells[0][2]+state.cells[0][3]
    if not positives or not negatives:
        raise UndefinedMetric("global positive and negative classes are both required")
    points = [(Fraction(fp,negatives),Fraction(tp,positives)) for _,fp,_,tp in reversed(state.cells)]
    if complete_origin and points[0]!=(0,0):
        points.insert(0,(Fraction(0),Fraction(0)))
    return integral(points),points


def estimate_budget(p):
    counts = p["rank_counts"]
    if (not isinstance(counts,list) or not 1<=len(counts)<=512 or
            any(not integer(n) or n>MAX_SAFE for n in counts) or not any(counts)):
        raise ValueError("one to 512 nonnegative exact rank counts, not all empty, required")
    if not integer(p["threshold_count"],2) or not integer(p["update_samples"]) or p["update_samples"]>max(counts):
        raise ValueError("at least two thresholds and update_samples within a declared rank total required")
    for key,allowed in (("prediction_bytes",(2,4,8)),("target_bytes",(1,4,8)),("threshold_bytes",(4,8))):
        if not integer(p[key],1) or p[key] not in allowed:
            raise ValueError(f"{key} has unsupported element width")
    total = sum(counts)
    raw_per_record = p["prediction_bytes"]+p["target_bytes"]
    result = dict(world_size=len(counts),global_samples=total,
                  largest_raw_state_bytes=max(counts)*raw_per_record,
                  gathered_raw_payload_bytes_per_rank=total*raw_per_record,
                  fixed_counter_state_bytes_per_rank=p["threshold_count"]*4*8,
                  threshold_vector_bytes_per_rank=p["threshold_count"]*p["threshold_bytes"],
                  source_update_path="vectorized" if p["update_samples"]<=50000 else "threshold loop",
                  selected_vectorized_int64_matrix_bytes=p["update_samples"]*p["threshold_count"]*8
                  if p["update_samples"]<=50000 else None,
                  source_counter_bytes=8,wire_bytes=None,peak_hbm_bytes=None,
                  native_metric_latency_ms=None)
    if any(isinstance(v,int) and v>MAX_SAFE for v in result.values()):
        raise ValueError("budget exceeds exact integer range")
    return result


def measure_latency():
    rows = []
    operations = 3
    for count in (16,64,256,1024):
        pairs = [(((i*37)%997)/996,i%2) for i in range(count)]
        methods = (lambda: exact_roc(pairs)[0],lambda: pairwise_auc(pairs))
        samples = [[],[]]
        for method in methods:
            for _ in range(2):
                method()
        for repeat in range(7):
            for index in ((0,1) if repeat%2==0 else (1,0)):
                begin = time.perf_counter_ns()
                for _ in range(operations):
                    methods[index]()
                samples[index].append((time.perf_counter_ns()-begin)/1e6/operations)
        rows.append(dict(samples=count,positive_negative_pairs=(count//2)**2,
                         primary_samples_ms=samples[0],baseline_samples_ms=samples[1],
                         primary_median_ms=statistics.median(samples[0]),
                         baseline_median_ms=statistics.median(samples[1]),
                         maximum_auc_error=abs(methods[0]()-methods[1]())))
    data = dict(provenance="measured",scope="Original rational grouped ROC versus pairwise scalar oracle",
                hardware=platform.machine(),processor=platform.processor(),python=platform.python_version(),
                warmups=2,repeats=7,operations_per_repeat=operations,alternating_order=True,
                statistic="median of seven repeat means; three operations per repeat",
                rows=rows,native_torchmetrics_measured=False)
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def simulate():
    shards = [[(.1,0),(.2,1)],[(.8,0),(.9,1)]]
    pairs = [row for shard in shards for row in shard]
    auc,points = exact_roc(pairs)
    thresholds = (0,.5,1)
    state = merge_states([threshold_counts(shard,thresholds) for shard in shards])
    coarse,coarse_points = threshold_roc(state)
    fine_grid = tuple(i/10 for i in range(11))
    fine = threshold_roc(threshold_counts(pairs,fine_grid))[0]
    tied = [(0.1,0),(0.2,1),(0.2,0),(0.9,1)]
    ones = [(1.0,0),(1.0,1)]
    endpoint_state = threshold_counts(ones,(0,1))
    endpoint_auc,endpoint_points = threshold_roc(endpoint_state)
    variants = []
    for grid in ((0,1),thresholds,fine_grid):
        value,_ = threshold_roc(threshold_counts(pairs,grid))
        variants.append(dict(thresholds=list(grid),count=len(grid),auc=value,error=value-auc))
    return dict(schema_version=1,round=111,provenance="synthetic",independently_reproduced=False,
                native_torchmetrics_measured=False,
                source=dict(pin=PIN,date="2026-09-08",license="Apache-2.0",
                            capability="Current supported Torch compatibility path; exact cat versus fixed-threshold SUM states are longstanding"),
                case=dict(shards=shards,positives=2,negatives=2,positive_negative_pairs=4,
                          exact_auc=auc,pairwise_oracle_auc=pairwise_auc(pairs),
                          local_aucs=[exact_roc(shard)[0] for shard in shards],
                          mean_local_auc=statistics.mean(exact_roc(shard)[0] for shard in shards),
                          roc_points=[[float(x),float(y)] for x,y in points],
                          coarse_thresholds=list(thresholds),coarse_cells=[list(row) for row in state.cells],
                          coarse_auc=coarse,fine_auc=fine,
                          coarse_roc_points=[[float(x),float(y)] for x,y in coarse_points]),
                ties=dict(pairs=tied,exact_auc=exact_roc(tied)[0],pairwise_auc=pairwise_auc(tied)),
                endpoint=dict(pairs=ones,thresholds=[0,1],exact_auc=exact_roc(ones)[0],
                              inspected_binned_rule_auc=endpoint_auc,
                              roc_points=[[float(x),float(y)] for x,y in endpoint_points],
                              origin_present=endpoint_points[0]==(0,0),
                              original_explicit_origin_completion_auc=threshold_roc(endpoint_state,True)[0]),
                variants=variants,budget=dict(input=DEFAULT,**estimate_budget(DEFAULT)),
                latency=json.loads((HERE/"latency.json").read_text(encoding="utf-8")))


def outputs():
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
            if not path.exists() or path.read_text(encoding="utf-8")!=content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 111 artifacts match." if args.check else "Wrote Round 111 artifacts.")


if __name__ == "__main__":
    main()
