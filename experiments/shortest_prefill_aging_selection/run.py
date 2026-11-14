"""Original rational rank/page ledger, not a native scheduler or wait guarantee."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
PIN="e5caad9a07573616e16614a9f374089959a7c48f"
REQUEST_FIELDS={"rid","prompt","output","matched","staged","arrival","entry","deferred"}
DEFAULT={"policy":"shortest-prefill-first","factor_numerator":1,"factor_denominator":2,
         "initial_clock":0,"events":[{"is_extend":True,"tokens":1000},
                                    {"is_extend":False,"tokens":1000},
                                    {"is_extend":False,"tokens":1000}],
         "requests":[{"rid":"old-long","prompt":1000,"output":0,"matched":0,"staged":0,
                      "arrival":0,"entry":0,"deferred":False},
                     {"rid":"new-short","prompt":10,"output":0,"matched":0,"staged":0,
                      "arrival":1000,"entry":1,"deferred":False}],
         "continuation":{"remaining":16384,"arrival":0},"budget":4096,"page":256}


def integer(v,maximum=2**30):
    return type(v) is int and 0<=v<=maximum


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original SPFclock/rank/page fields required")
    if p["policy"]!="shortest-prefill-first":
        raise ValueError("original prototype models SPFonly;nonzero native aging requires SPF")
    if not integer(p["factor_numerator"],64) or not integer(p["factor_denominator"],1024) or not p["factor_denominator"]:
        raise ValueError("bounded nonnegative rational factor required;native default0remains opt-in")
    if not integer(p["initial_clock"]) or not integer(p["budget"]) or not integer(p["page"]) or not p["page"]:
        raise ValueError("nonnegative clock/budget and positive page required")
    if not isinstance(p["events"],list) or len(p["events"])>128:
        raise ValueError("bounded original launch event list required")
    for e in p["events"]:
        if not isinstance(e,dict) or set(e)!={"is_extend","tokens"} or type(e["is_extend"]) is not bool or not integer(e["tokens"]):
            raise ValueError("explicit native-is_extend witness and token count required")
    if not isinstance(p["requests"],list) or len(p["requests"])>128:
        raise ValueError("bounded original request ledger required")
    names=[]
    for r in p["requests"]:
        if not isinstance(r,dict) or set(r)!=REQUEST_FIELDS or not isinstance(r["rid"],str) or not r["rid"]:
            raise ValueError("exact request fields and named identity required")
        if type(r["deferred"]) is not bool or any(not integer(r[k]) for k in REQUEST_FIELDS-{"rid","deferred"}):
            raise ValueError("bounded request integers and duplicate-deferral witness required")
        names.append(r["rid"])
    if len(set(names))!=len(names):
        raise ValueError("unique request identities required")
    c=p["continuation"]
    if not isinstance(c,dict) or set(c)!={"remaining","arrival"} or any(not integer(v) for v in c.values()):
        raise ValueError("explicit continuation remaining/arrival required")


def work(r):
    return max(1,r["prompt"]+r["output"]-max(r["matched"],r["staged"]))


def rank_numerator(r,clock,num,den):
    return work(r)*den-num*max(0,clock-r["arrival"])


def plan(p,legacy_clock=False,without_aging=False):
    validate(p)
    clock=p["initial_clock"]+sum(e["tokens"] for e in p["events"] if legacy_clock or e["is_extend"])
    num=0 if without_aging else p["factor_numerator"];den=p["factor_denominator"]
    ordered=sorted(p["requests"],key=lambda r:(r["deferred"],rank_numerator(r,clock,num,den),r["entry"]))
    ranks=[{"rid":r["rid"],"work":work(r),"waited_prefill_tokens":max(0,clock-r["arrival"]),
            "rank_numerator":rank_numerator(r,clock,num,den),"rank_denominator":den} for r in ordered]
    c=p["continuation"];active=c["remaining"]*den-num*max(0,clock-c["arrival"])
    reserved=0;reservations=[];limit=None
    if p["budget"]>=2*p["page"]:
        for r in ordered:
            charge=(work(r)+p["page"]-1)//p["page"]*p["page"]
            if rank_numerator(r,clock,num,den)>=active or reserved+charge>p["budget"]-p["page"]:
                break
            reserved+=charge;reservations.append({"rid":r["rid"],"work":work(r),"page_charge":charge})
        if reserved:
            limit=(p["budget"]-reserved)//p["page"]*p["page"]
    return {"processed_prefill_clock":clock,"ordered_rids":[r["rid"] for r in ordered],"ranks":ranks,
            "continuation_rank_numerator":active,"reserved_tokens":reserved,"reservations":reservations,
            "continuation_limit":limit,"native_schedule_verified":None,"wall_clock_wait_bound_seconds":None}


def affine_order(rows,clock,num,den):
    if any(r["arrival"]>clock for r in rows):
        raise ValueError("common-clock cancellation requires all requests already arrived")
    return sorted(rows,key=lambda r:(r["deferred"],work(r)*den+num*r["arrival"],r["entry"]))


def measure():
    data=[]
    for n in (64,256,1024,4096):
        rows=[{**DEFAULT["requests"][0],"rid":str(i),"prompt":32+i%1024,
               "arrival":(i*17)%4096,"entry":i,"deferred":i%11==0} for i in range(n)]
        def adjusted():return [r["rid"] for r in affine_order(rows,4096,1,2)]
        def full():return [r["rid"] for r in sorted(rows,key=lambda r:(r["deferred"],rank_numerator(r,4096,1,2),r["entry"]))]
        funcs=(adjusted,full)
        if adjusted()!=full():raise RuntimeError("clock-cancelled order differs")
        for _ in range(2):
            for f in funcs:f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                start=time.perf_counter_ns()
                for _ in range(3):funcs[i]()
                samples[i].append((time.perf_counter_ns()-start)/3/1e6)
        data.append({"requests":n,"adjusted_samples_ms":samples[0],"full_samples_ms":samples[1],
                     "adjusted_median_ms":statistics.median(samples[0]),"full_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,
            "repeats":7,"operations_per_sample":3,"method":"original arrived-request rational sorts,common-clock adjusted versus full rank",
            "excludes":"setup,native scheduler/cache matching/prefetch/admission,GPUs,queue latency or throughput","rows":data}


def build_report(latency):
    fresh={**DEFAULT["requests"][1],"arrival":3000}
    primary={**DEFAULT,"events":[{"is_extend":True,"tokens":3000}],"requests":[DEFAULT["requests"][0],fresh]}
    variants=[("new_arrival",primary),("fixed_backlog",DEFAULT),
              ("deferred_old",{**primary,"requests":[{**primary["requests"][0],"deferred":True},fresh]}),
              ("aging_disabled",{**primary,"factor_numerator":0}),
              ("page_charge",{**DEFAULT,"events":[],"factor_numerator":0,
                              "requests":[{**DEFAULT["requests"][0],"prompt":257}]}),
              ("insufficient_budget",{**primary,"budget":511})]
    matrix=[{"name":name,"inputs":p,"outputs":plan(p)} for name,p in variants]
    return {"schema_version":1,"round":161,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"sgl-project/sglang","pin":PIN,"license":"Apache-2.0 rights retained,no implementation mirror",
                      "chronology":"actualNov14UTC01:50:59/BJT09:50:59 opt-in SPFaging addition and stale decode counter correction",
                      "verified":"work=max1(total-max matched/staged);rank=work-factor*max0(P-arrival);deferred before rank;page reservation compares aged ranks;only extend launches charge counter",
                      "scope":"original rational clock/rank/page ledger,not native matching,prefetch,admission or wall-time fairness"},
            "case":{"inputs":primary,"aged":plan(primary),"unaged":plan(primary,without_aging=True)},
            "matrix":matrix,"clock_case":{"correct":plan(DEFAULT),"legacy":plan(DEFAULT,legacy_clock=True)},
            "latency":latency,"native_scheduler_verified":None,"wall_clock_wait_bound_seconds":None,
            "browser_visual_acceptance":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.measure:(HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text()))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:raise SystemExit(f"stale artifact:{file}")
        else:file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round161 sealed artifacts match" if a.check else "Wrote Round161original artifacts")


if __name__=="__main__":main()
