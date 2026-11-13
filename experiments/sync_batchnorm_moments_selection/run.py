"""Original count-aware scalar normalization; no native distributed execution."""

import argparse
import json
import math
import platform
import statistics
import struct
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "e5e96ce34777fb56e66232669a221c51955a57ef"
MAX_SAFE = 2**53-1
DEFAULT = dict(rank_counts=[256,256,256,256],channels=1024,stat_bytes=4,value_bytes=4)


def integer(value,minimum=0):
    return isinstance(value,int) and not isinstance(value,bool) and value>=minimum


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


@dataclass(frozen=True)
class Moments:
    count: int
    mean: float | None
    m2: float

    def __post_init__(self):
        if not integer(self.count) or not finite(self.m2) or self.m2<0:
            raise ValueError("moments require a nonnegative integer count and finite nonnegative M2")
        if self.count==0:
            if self.mean is not None or self.m2!=0:
                raise ValueError("empty summary has undefined mean and zero additive M2")
        elif not finite(self.mean):
            raise ValueError("nonempty summary requires a finite mean")

    @property
    def population_variance(self):
        return self.m2/self.count if self.count else None

    @property
    def unbiased_variance(self):
        return self.m2/(self.count-1) if self.count>1 else None


def summarize(values):
    count,mean,m2 = 0,0.0,0.0
    for value in values:
        if not finite(value):
            raise ValueError("observations must be finite scalar numbers")
        count += 1
        delta = value-mean
        mean += delta/count
        m2 += delta*(value-mean)
    return Moments(count,mean if count else None,m2)


def merge(a,b):
    if not isinstance(a,Moments) or not isinstance(b,Moments):
        raise TypeError("merge requires two validated moment summaries")
    if not a.count:
        return b
    if not b.count:
        return a
    n = a.count+b.count
    delta = b.mean-a.mean
    return Moments(n,a.mean+delta*b.count/n,a.m2+b.m2+delta*delta*a.count*b.count/n)


def merged(summaries):
    result = Moments(0,None,0.0)
    for summary in summaries:
        result = merge(result,summary)
    return result


def exact_reference(values):
    if not values:
        raise ValueError("exact reference requires observations")
    fractions = [Fraction(value) for value in values]
    mean = sum(fractions)/len(values)
    m2 = sum((x-mean)**2 for x in fractions)
    return dict(count=len(values),mean=float(mean),m2=float(m2),
                population_variance=float(m2/len(values)),
                unbiased_variance=float(m2/(len(values)-1)) if len(values)>1 else None)


def forward(values,gamma=1.5,beta=-.25,epsilon=.1):
    if not finite(epsilon) or epsilon<=0 or not finite(gamma) or not finite(beta):
        raise ValueError("finite affine parameters and positive finite epsilon required")
    stats = summarize(values)
    if stats.count==0:
        raise ValueError("global all-empty normalization is undefined")
    inverse = 1/math.sqrt(stats.population_variance+epsilon)
    return [gamma*(x-stats.mean)*inverse+beta for x in values],stats,inverse


def gradients(shards,dy_shards,gamma=1.5,epsilon=.1,local_only=False):
    if len(shards)!=len(dy_shards) or not shards:
        raise ValueError("matching nonempty rank collections required")
    if any(len(x)!=len(dy) for x,dy in zip(shards,dy_shards)):
        raise ValueError("each rank needs one upstream gradient per observation")
    values = [x for shard in shards for x in shard]
    dy = [v for shard in dy_shards for v in shard]
    if any(not finite(v) for v in dy):
        raise ValueError("upstream gradients must be finite")
    _,stats,inverse = forward(values,gamma=gamma,epsilon=epsilon)
    sums = [(math.fsum(d),math.fsum(g*(x-stats.mean) for x,g in zip(xs,d)))
            for xs,d in zip(shards,dy_shards)]
    global_sums = (math.fsum(row[0] for row in sums),math.fsum(row[1] for row in sums))
    dx = []
    for xs,d,rank_sums in zip(shards,dy_shards,sums):
        sum_dy,sum_dy_xmu = rank_sums if local_only else global_sums
        dx += [gamma*inverse*(g-sum_dy/stats.count-
                             (x-stats.mean)*inverse*inverse*sum_dy_xmu/stats.count)
               for x,g in zip(xs,d)]
    return dict(dx=dx,dgamma=math.fsum(g*(x-stats.mean)*inverse for x,g in zip(values,dy)),
                dbeta=math.fsum(dy),sum_dy=global_sums[0],sum_dy_xmu=global_sums[1])


def finite_difference(values,dy,gamma=1.5,beta=-.25,epsilon=.1):
    step = 1e-5
    def loss(xs,g,b):
        output,_,_ = forward(xs,g,b,epsilon)
        return math.fsum(y*d for y,d in zip(output,dy))
    result = []
    for i in range(len(values)):
        plus,minus = list(values),list(values)
        plus[i] += step
        minus[i] -= step
        result.append((loss(plus,gamma,beta)-loss(minus,gamma,beta))/(2*step))
    return dict(dx=result,dgamma=(loss(values,gamma+step,beta)-loss(values,gamma-step,beta))/(2*step),
                dbeta=(loss(values,gamma,beta+step)-loss(values,gamma,beta-step))/(2*step),step=step)


def estimate_budget(p):
    counts = p["rank_counts"]
    if (not isinstance(counts,list) or not 1<=len(counts)<=512 or
            any(not integer(n) or n>MAX_SAFE for n in counts) or not any(counts)):
        raise ValueError("one to 512 nonnegative exact rank counts, not all empty, required")
    if not integer(p["channels"],1) or p["channels"]>MAX_SAFE:
        raise ValueError("positive exact integer channel count required")
    if (not integer(p["stat_bytes"],1) or p["stat_bytes"] not in (4,8) or
            not integer(p["value_bytes"],1) or p["value_bytes"] not in (1,2,4,8)):
        raise ValueError("stat_bytes 4/8 and value_bytes 1/2/4/8 required")
    c,w = p["channels"],len(counts)
    payload = (2*c+1)*p["stat_bytes"]
    exact_count = (all(struct.unpack("<f",struct.pack("<f",n))[0]==n for n in counts)
                   if p["stat_bytes"]==4 else True)
    result = dict(world_size=w,total_observations_per_channel=sum(counts),
                  nonempty_ranks=sum(n>0 for n in counts),
                  forward_packet_bytes_per_rank=payload,
                  forward_gather_buffer_bytes_per_rank=w*payload,
                  backward_reduction_payload_bytes_per_rank=2*c*p["stat_bytes"],
                  hypothetical_feature_gather_bytes_per_rank=sum(counts)*c*p["value_bytes"],
                  all_packet_counts_exact_in_selected_float=exact_count,
                  all_saved_rank_counts_fit_int32=all(n<=2**31-1 for n in counts),
                  native_collective_latency_ms=None,wire_bytes=None,peak_hbm_bytes=None)
    if any(isinstance(v,int) and not isinstance(v,bool) and v>MAX_SAFE for v in result.values()):
        raise ValueError("budget exceeds exact integer range")
    return result


def measure_latency():
    rows = []
    operations = 10
    for count in (16,64,256,1024):
        shards = [[float((rank*count+i)%31-15) for i in range(count)] for rank in range(4)]
        summaries = [summarize(xs) for xs in shards]
        values = [x for xs in shards for x in xs]
        def rescan():
            mean = math.fsum(values)/len(values)
            return math.fsum((x-mean)**2 for x in values)/len(values)
        methods = (lambda: merged(summaries).population_variance,rescan)
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
        rows.append(dict(observations_per_rank=count,world_size=4,
                         primary_samples_ms=samples[0],baseline_samples_ms=samples[1],
                         primary_median_ms=statistics.median(samples[0]),
                         baseline_median_ms=statistics.median(samples[1]),
                         population_variance_error=abs(methods[0]()-methods[1]())))
    data = dict(provenance="measured",scope="Original precomputed summary merge versus feature two-pass rescan",
                hardware=platform.machine(),processor=platform.processor(),python=platform.python_version(),
                warmups=2,repeats=7,operations_per_repeat=operations,alternating_order=True,
                statistic="median of seven repeat means; ten operations per repeat",
                excluded="Initial local summaries, data movement, network and native kernels",rows=rows,
                native_pytorch_measured=False)
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def simulate():
    shards = [[-3.0],[1.0,7.0,9.0],[],[10.0,14.0]]
    dy = [[.2],[-.4,.7,1.1],[],[-.3,.5]]
    summaries = [summarize(xs) for xs in shards]
    stats = merged(summaries)
    nonempty = [s for s in summaries if s.count]
    values = [x for xs in shards for x in xs]
    upstream = [g for gs in dy for g in gs]
    reference = exact_reference(values)
    correct = gradients(shards,dy)
    wrong = gradients(shards,dy,local_only=True)
    fd = finite_difference(values,upstream)
    large = [1e12+x for x in (-3,-1,1,3)]
    stable = merged([summarize(large[:2]),summarize(large[2:])])
    naive_mean = math.fsum(large)/len(large)
    naive_var = math.fsum(x*x for x in large)/len(large)-naive_mean*naive_mean
    return dict(schema_version=1,round=110,provenance="synthetic",independently_reproduced=False,
                native_pytorch_measured=False,
                source=dict(pin=PIN,date="2026-07-29",wrapper_last_change="2026-06-04",
                            license="PyTorch LICENSE; GitHub SPDX NOASSERTION",
                            capability="Count-aware SyncBatchNorm collectives and merged statistics; longstanding mechanics"),
                moments=dict(shards=shards,rank_counts=[len(xs) for xs in shards],count=stats.count,
                             mean=stats.mean,m2=stats.m2,population_variance=stats.population_variance,
                             unbiased_variance=stats.unbiased_variance,exact_reference=reference,
                             mean_of_rank_means=math.fsum(s.mean for s in nonempty)/len(nonempty),
                             mean_of_rank_variances=math.fsum(s.population_variance for s in nonempty)/len(nonempty),
                             count_weighted_local_variances=math.fsum(s.m2 for s in nonempty)/stats.count,
                             declared_collective_participants=list(range(len(shards))),
                             empty_ranks=[i for i,xs in enumerate(shards) if not xs]),
                gradient=dict(upstream=dy,analytic=correct,finite_difference=fd,wrong_local_sums=wrong,
                              maximum_dx_finite_difference_error=max(abs(a-b) for a,b in zip(correct["dx"],fd["dx"])),
                              maximum_wrong_local_dx_error=max(abs(a-b) for a,b in zip(correct["dx"],wrong["dx"]))),
                cancellation=dict(offset=1e12,values=large,exact_variance=exact_reference(large)["population_variance"],
                                  merged_variance=stable.population_variance,raw_second_moment_variance=naive_var),
                budget=dict(input=DEFAULT,**estimate_budget(DEFAULT)),
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
    print("Round 110 artifacts match." if args.check else "Wrote Round 110 artifacts.")


if __name__ == "__main__":
    main()
