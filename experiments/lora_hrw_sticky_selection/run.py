"""Original score-table LoRA placement model;no native BLAKE3 reproduction."""

import argparse
import heapq
from itertools import combinations
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="2cac4d767f49b5b000c0f354fe369554744bba2b"
DEFAULT={"rank":16,"input_width":4096,"output_width":4096,"selected_modules":32,"element_bytes":2,"replicas":2}
WORKERS=[
    {"id":7,"dp":0,"score":100,"used":0,"capacity":2},
    {"id":7,"dp":1,"score":90,"used":0,"capacity":2},
    {"id":9,"dp":0,"score":80,"used":0,"capacity":2},
    {"id":10,"dp":0,"score":70,"used":0,"capacity":2},
]


def identity(worker):
    return (worker["id"],worker["dp"])


def validate(workers,replicas,prior):
    if type(replicas) is not int or replicas<0:
        raise ValueError("nonnegative exact replica factor required")
    seen=set()
    for w in workers:
        if set(w)!={"id","dp","score","used","capacity"}:
            raise ValueError("worker fields must match score-table contract")
        if any(type(w[k]) is not int or not 0<=w[k]<=2**53-1 for k in ("id","dp","score")):
            raise ValueError("bounded exact worker identity and score required")
        if (w["used"] is None)!=(w["capacity"] is None):
            raise ValueError("slot usage and capacity must both be unknown or known")
        if w["used"] is not None and any(type(w[k]) is not int or w[k]<0 or w[k]>2**53-1 for k in ("used","capacity")):
            raise ValueError("invalid bounded slot data")
        if identity(w) in seen:
            raise ValueError("duplicate worker and dp-rank identity")
        seen.add(identity(w))
    for p in prior:
        if len(p)!=2 or any(type(v) is not int or v<0 or v>2**53-1 for v in p):
            raise ValueError("prior identities must contain bounded worker/dp pair")


def ranked(workers):
    return sorted(workers,key=lambda w:(-w["score"],w["id"],w["dp"]))


def full(worker):
    return worker["capacity"] is not None and worker["used"]>=worker["capacity"]


def allocate(workers,replicas,prior=(),sticky=True,slot_aware=True):
    validate(workers,replicas,prior)
    order=ranked(workers)
    if not order or replicas==0:
        return {"selected":[],"fallback":None,"selected_full_count":0,"unknown_slot_workers":sum(w["capacity"] is None for w in workers)}
    if not slot_aware:
        selected=order[:replicas];fallback=None
    else:
        available=[w for w in order if not full(w)]
        if not available:
            selected=order[:replicas];fallback="all_workers_full_basic_hrw"
        else:
            previous={tuple(p) for p in prior} if sticky else set()
            keep=[w for w in available if identity(w) in previous]
            remaining=[w for w in available if identity(w) not in previous]
            selected=(keep+remaining)[:replicas];fallback=None
    return {"selected":[list(identity(w)) for w in selected],"fallback":fallback,
            "selected_full_count":sum(full(w) for w in selected),
            "unknown_slot_workers":sum(w["capacity"] is None for w in workers)}


def independent_subset_oracle(workers,replicas,prior=(),sticky=True):
    validate(workers,replicas,prior)
    order=ranked(workers);available=[w for w in order if not full(w)]
    if not available:
        return [list(identity(w)) for w in order[:replicas]]
    previous={tuple(p) for p in prior} if sticky else set()
    size=min(replicas,len(available));ranks={identity(w):i for i,w in enumerate(order)}
    candidates=list(combinations(available,size))
    chosen=min(candidates,key=lambda subset:(-sum(identity(w) in previous for w in subset),tuple(sorted(ranks[identity(w)] for w in subset))))
    return [list(identity(w)) for w in sorted(chosen,key=lambda w:(identity(w) not in previous,ranks[identity(w)]))]


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int or v<1 for v in p.values()) or p["element_bytes"] not in (1,2,4,8):
        raise ValueError("positive exact dimensions and selected element width required")
    one=p["rank"]*(p["input_width"]+p["output_width"])*p["selected_modules"]*p["element_bytes"]
    out={"selected_adapter_ab_payload_bytes":one,"selected_replica_payload_bytes":one*p["replicas"]}
    if max(*p.values(),*out.values())>2**53-1:
        raise ValueError("exact browser budget range exceeded")
    return out


def timing_paths(n):
    workers=[{"id":i,"dp":i%4,"score":((i+1)*2654435761)%2**32,"used":0,"capacity":2} for i in range(n)]
    key=lambda w:(w["score"],-w["id"],-w["dp"])
    def topk():
        return [identity(w) for w in heapq.nlargest(4,workers,key=key)]
    def sorting():
        return [identity(w) for w in sorted(workers,key=key,reverse=True)[:4]]
    return topk,sorting


def measure():
    rows=[]
    for n in (32,128,512,2048):
        functions=timing_paths(n);assert functions[0]()==functions[1]()
        for _ in range(2):
            for f in functions:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    functions[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"workers":n,"replicas":4,"topk_samples_ms":samples[0],"sort_samples_ms":samples[1],
                     "topk_median_ms":statistics.median(samples[0]),"sort_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"original precomputed score-table top4 heap versus full sort",
            "excludes":"hashing,slot/sticky policy,controller,Rust,Dynamo,adapter transfer/load,inference,GPU","rows":rows}


def build_report(latency):
    prior=[[9,0],[10,0]]
    natural=allocate(WORKERS,2,sticky=False)
    sticky=allocate(WORKERS,2,prior)
    retained=lambda result:len({tuple(w) for w in result["selected"]}&{tuple(w) for w in prior})
    all_full=[{**w,"used":2} for w in WORKERS]
    one_free=[{**w,"used":0 if w["id"]==10 else 2} for w in WORKERS]
    lost_prior=[{**w,"used":2 if w["id"]==9 else 0} for w in WORKERS]
    added_high=WORKERS+[{"id":11,"dp":0,"score":110,"used":0,"capacity":2}]
    tie=[{**w,"score":100} for w in WORKERS]
    return {"schema_version":1,"round":132,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"ai-dynamo/dynamo","pin":PIN,"license":"Apache-2.0 in inspected files",
                      "tree_date_utc":"2026-11-14T00:51:42Z","chronology":"latest tree update is sidebar CSS;Nov6crate extraction and Nov11crate refactor are not HRW invention",
                      "native_hash":"BLAKE3 over LoRA name,worker_id LE,dp_rank LE;first8bytes little-endian u64",
                      "model_hash":"none;explicit original bounded score tables,not native BLAKE3 bit parity"},
            "case":{"workers":WORKERS,"replica_factor":2,"prior":prior,
                    "natural":natural,"sticky":sticky,
                    "sticky_prior_full":allocate(lost_prior,2,prior),"only_one_nonfull":allocate(one_free,2,prior),
                    "all_full":allocate(all_full,2,prior),"added_high_basic":allocate(added_high,2,sticky=False),
                    "tie_order":[list(identity(w)) for w in ranked(tie)],
                    "tie_reversed_order":[list(identity(w)) for w in ranked(list(reversed(tie)))],
                    "oracle_sticky":independent_subset_oracle(WORKERS,2,prior),
                    "natural_prior_retained":retained(natural),"sticky_prior_retained":retained(sticky),
                    "missing_slots_policy":"unknown treated eligible in pinned allocator,not proven free",
                    "not_modeled":"hash distribution,request routing,admission,eviction,actual residency,global MCF/controller"},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_dynamo_latency_ms":None,"hash_distribution_quality":None,"adapter_load_latency_ms":None,"native_peak_hbm_bytes":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for path,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if a.check:
            if path.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact: {path}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
    print("Round132 sealed artifacts match" if a.check else "Wrote Round132 original artifacts")


if __name__=="__main__":
    main()
