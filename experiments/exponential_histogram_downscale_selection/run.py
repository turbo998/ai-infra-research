"""Original exact rational exponential buckets; no native OpenTelemetry execution."""

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import platform
import statistics
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "63c5f44275f152fa0a9629463db43474fe7214e2"
MAX_EXACT = 2**53-1
DEFAULT = {"streams":10000,"live_states_per_stream":2,"max_size":160,
           "positive_span":160,"negative_span":17}


def valid_scale(scale):
    if type(scale) is not int or not -10<=scale<=0:
        raise ValueError("original exact model supports integer scales -10 through 0")


def exact_value(value):
    if type(value) is not int and not isinstance(value,Fraction):
        raise ValueError("exact integer/Fraction observation required; floats are not normalized")
    value = Fraction(value)
    if value<0:
        raise ValueError("original model supports nonnegative observations only")
    return value


def power_two(exponent):
    return Fraction(2**exponent) if exponent>=0 else Fraction(1,2**(-exponent))


def bucket_index(value,scale=0):
    valid_scale(scale)
    value = exact_value(value)
    if not value:
        raise ValueError("zero has a separate counter, not a positive bucket index")
    exponent = value.numerator.bit_length()-value.denominator.bit_length()
    ceil_log = exponent if value<=power_two(exponent) else exponent+1
    return (ceil_log-1)//(2**(-scale))


def bucket_bounds(index,scale):
    valid_scale(scale)
    if type(index) is not int:
        raise ValueError("integer bucket index required")
    width = 2**(-scale)
    return power_two(index*width),power_two((index+1)*width)


@dataclass(frozen=True)
class Histogram:
    scale: int
    counts: tuple[tuple[int,int], ...]
    zero_count: int = 0

    def __post_init__(self):
        valid_scale(self.scale)
        if type(self.zero_count) is not int or self.zero_count<0:
            raise ValueError("nonnegative zero count required")
        if not isinstance(self.counts,tuple):
            raise ValueError("immutable bucket entries required")
        last = None
        for entry in self.counts:
            if not isinstance(entry,tuple) or len(entry)!=2:
                raise ValueError("immutable (index,count) entries required")
            index,count = entry
            if type(index) is not int or type(count) is not int or count<=0 or (last is not None and index<=last):
                raise ValueError("strictly increasing indices with positive integer counts required")
            last = index

    @property
    def count(self):
        return self.zero_count+sum(count for _,count in self.counts)

    @property
    def span(self):
        return self.counts[-1][0]-self.counts[0][0]+1 if self.counts else 0


def histogram(values,scale=0):
    valid_scale(scale)
    counts,zeros = Counter(),0
    for raw in values:
        value = exact_value(raw)
        if value==0:
            zeros += 1
        else:
            counts[bucket_index(value,scale)] += 1
    return Histogram(scale,tuple(sorted(counts.items())),zeros)


def downscale(hist,target):
    valid_scale(target)
    if target>hist.scale:
        raise ValueError("cannot recover finer buckets from coarsened counts")
    factor = 2**(hist.scale-target)
    counts = Counter()
    for index,count in hist.counts:
        counts[index//factor] += count
    return Histogram(target,tuple(sorted(counts.items())),hist.zero_count)


def merge(states,target=None):
    states = tuple(states)
    if not states:
        raise ValueError("at least one declared histogram state required")
    target = min(state.scale for state in states) if target is None else target
    valid_scale(target)
    counts,zeros = Counter(),0
    for state in states:
        aligned = downscale(state,target)
        counts.update(dict(aligned.counts))
        zeros += aligned.zero_count
    return Histogram(target,tuple(sorted(counts.items())),zeros)


def fit(hist,max_size):
    if type(max_size) is not int or max_size<1:
        raise ValueError("positive maximum logical span required")
    for target in range(hist.scale,-11,-1):
        candidate = downscale(hist,target)
        if candidate.span<=max_size:
            return candidate
    raise ValueError("requested span cannot fit the supported scale range; no observation dropped")


def quantile_interval(hist,q):
    if type(q) is not int and not isinstance(q,Fraction):
        raise ValueError("exact quantile fraction required")
    q = Fraction(q)
    if not 0<q<=1 or not hist.count:
        raise ValueError("nearest-rank quantile requires 0<q<=1 and nonempty data")
    rank = (q.numerator*hist.count+q.denominator-1)//q.denominator
    if rank<=hist.zero_count:
        return {"rank":rank,"lower":Fraction(0),"upper":Fraction(0),"zero":True}
    cumulative = hist.zero_count
    for index,count in hist.counts:
        cumulative += count
        if cumulative>=rank:
            lower,upper = bucket_bounds(index,hist.scale)
            return {"rank":rank,"lower":lower,"upper":upper,"zero":False}
    raise AssertionError("validated counts failed rank lookup")


def exact_quantile(values,q):
    ordered = sorted(exact_value(value) for value in values)
    q = Fraction(q)
    if not ordered or not 0<q<=1:
        raise ValueError("undefined nearest-rank quantile")
    rank = (q.numerator*len(ordered)+q.denominator-1)//q.denominator
    return ordered[rank-1]


def interval_contains(interval,value):
    return value==0 if interval["zero"] else interval["lower"]<value<=interval["upper"]


def capacity(span):
    return 0 if span==0 else 1<<(span-1).bit_length()


def estimate_budget(p):
    for key in DEFAULT:
        value = p[key]
        minimum = 0 if key in ("positive_span","negative_span") else 1
        if type(value) is not int or not minimum<=value<=MAX_EXACT:
            raise ValueError(f"{key} requires a bounded exact integer >= {minimum}")
    if max(p["positive_span"],p["negative_span"])>p["max_size"]:
        raise ValueError("logical bucket span exceeds selected max_size")
    pos,neg = capacity(p["positive_span"]),capacity(p["negative_span"])
    per_state = (pos+neg+1)*8
    result = {"positive_fresh_capacity":pos,"negative_fresh_capacity":neg,
              "selected_logical_counter_bytes_per_state":(p["positive_span"]+p["negative_span"]+1)*8,
              "selected_fresh_counter_bytes_per_state":per_state,
              "selected_live_counter_payload_bytes":per_state*p["streams"]*p["live_states_per_stream"]}
    if any(value>MAX_EXACT for value in result.values()):
        raise ValueError("budget exceeds exact integer range")
    return {**result,"native_retained_capacity_bytes":None,"total_sdk_memory_bytes":None,
            "wire_bytes":None,"native_collection_latency_ms":None}


def fixture():
    return [Fraction(value) for value in (0,Fraction(1,4),Fraction(1,2),1,2,3,4,8,16,32)]


def measure_latency():
    records = []
    for per_shard in (32,128,512,2048):
        shards = [[Fraction(0) if j%13==0 else Fraction(2**((j+i)%7),4)
                   for j in range(per_shard)] for i in range(8)]
        states = [histogram(values,0 if i%2==0 else -1) for i,values in enumerate(shards)]
        raw = [value for values in shards for value in values]
        primary = lambda:merge(states,-2)
        baseline = lambda:histogram(raw,-2)
        assert primary()==baseline()
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
        records.append({"shards":8,"samples_per_shard":per_shard,"total_observations":len(raw),
                        "result_count_error":0,"bucket_state_equal":True,
                        "primary_samples_ms":samples["primary"],"baseline_samples_ms":samples["baseline"],
                        "primary_median_ms":statistics.median(samples["primary"]),
                        "baseline_median_ms":statistics.median(samples["baseline"])})
    record = {"provenance":"measured","independently_reproduced":True,
              "implementation":"Original immutable precomputed-state alignment/merge versus exact raw-observation replay",
              "hardware":platform.processor() or platform.machine(),"python":platform.python_version(),
              "warmups":2,"repeats":7,"operations_per_sample":3,"order":"alternating",
              "exclusions":["initial per-shard histograms","native circular buffers","OpenTelemetry SDK collection",
                            "serialization","network","concurrent record/merge"],
              "rows":records}
    (HERE/"latency.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")


def simulate():
    values = fixture()
    original = histogram(values)
    bounded = fit(original,4)
    zero = Histogram(0,(),0)
    aligned_merge = merge((histogram(values[:5]),histogram(values[5:],-1),zero),bounded.scale)
    wrong = Counter()
    divisor = 2**(-bounded.scale)
    for index,count in original.counts:
        truncated = index//divisor if index>=0 else -((-index)//divisor)
        wrong[truncated] += count
    wrong_hist = Histogram(bounded.scale,tuple(sorted(wrong.items())),original.zero_count)
    variants = []
    for name,scale,use_truncation in (("Original scale",0,False),("Coarser scale",-1,False),
                                      ("Fit four logical slots",bounded.scale,False),
                                      ("Wrong signed truncation",bounded.scale,True)):
        state = wrong_hist if use_truncation else downscale(original,scale)
        violations = 0
        for value in values:
            if not value:
                continue
            old_index = bucket_index(value)
            index = old_index//(2**(-scale)) if not use_truncation else (
                old_index//divisor if old_index>=0 else -((-old_index)//divisor))
            lower,upper = bucket_bounds(index,scale)
            violations += not lower<value<=upper
        variants.append({"name":name,"scale":scale,"span":state.span,"count":state.count,
                         "containment_violations":violations,
                         "containment_percent":100*(len(values)-violations)/len(values),
                         "change_points":-100*violations/len(values)})
    quantiles = []
    for q in (Fraction(1,2),Fraction(9,10),Fraction(99,100)):
        interval = quantile_interval(bounded,q)
        actual = exact_quantile(values,q)
        quantiles.append({"q":str(q),"rank":interval["rank"],"exact":str(actual),
                          "lower_exclusive":str(interval["lower"]),"upper_inclusive":str(interval["upper"]),
                          "contains_exact":interval_contains(interval,actual)})
    return {"round":113,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"open-telemetry/opentelemetry-go","pin":PIN,
                      "date":"2026-11-09T19:32:45Z","license":"Apache-2.0",
                      "paths":["sdk/metric/internal/aggregate/exponential_histogram.go",
                               "sdk/metric/metricdata/data.go","sdk/metric/aggregation.go"]},
            "case":{"observations":[str(value) for value in values],"count":original.count,
                    "zero_count":original.zero_count,"original_scale":original.scale,
                    "original_counts":original.counts,"original_span":original.span,
                    "original_nonempty_buckets":len(original.counts),
                    "max_size":4,"fitted_scale":bounded.scale,"fitted_counts":bounded.counts,
                    "fitted_span":bounded.span,"fitted_count":bounded.count,
                    "aligned_merge_equal":aligned_merge==bounded,
                    "wrong_truncation_counts":wrong_hist.counts,"wrong_count":wrong_hist.count,
                    "wrong_containment_violations":variants[-1]["containment_violations"]},
            "quantiles":quantiles,"variants":variants,"budget":{"input":DEFAULT,**estimate_budget(DEFAULT)},
            "latency":json.loads((HERE/"latency.json").read_text(encoding="utf-8")),
            "unknowns":{"native_latency_ms":None,"total_sdk_memory_bytes":None,"calibrated_confidence":None}}


def outputs():
    from experiments.exponential_histogram_downscale_selection.render import render_outputs
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
    print("Round 113 artifacts match." if args.check else "Wrote Round 113 artifacts.")


if __name__=="__main__":
    main()
