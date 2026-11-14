"""Original integer chiplet row ownership and one-round candidate search."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PIN="2aa7eb3ea5cea8c99f484ca726be2f4d9c0ba418"
DEFAULT={"requests":10,"query_rows":8,"chiplets":8,"cus":256,"topk":16,"kv_heads":1,"explicit_splits":None}


def integer(x,lo,hi):return type(x) is int and lo<=x<=hi


def ownership(requests,q,x,placement):
    if placement=="request":
        return [q*(requests//x+(i<requests%x)) for i in range(x)],((requests+x-1)//x)*x*q
    full=(requests//x)*q;tail=(requests%x)*q;per=(tail+x-1)//x
    return [full+max(0,min(per,tail-i*per)) for i in range(x)],(full+per)*x


def enumerate_ownership(requests,q,x,placement):
    rows=requests*q;counts=[0]*x
    full=(requests//x)*q;tail=rows-full*x;per=(tail+x-1)//x
    grid=((requests+x-1)//x)*x*q if placement=="request" else (full+per)*x
    mapped=[]
    for block in range(grid):
        chip=block%x;slot=block//x
        row=((slot//q)*x+chip)*q+slot%q if placement=="request" or slot<full else full*x+chip*per+slot-full
        mapped.append(row)
        if row<rows:counts[chip]+=1
    return counts,grid,mapped


def plan(p,strategy="compare"):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):raise ValueError("Exact original topology fields required")
    limits={"requests":(0,64),"query_rows":(1,32),"chiplets":(0,16),"cus":(1,4096),"topk":(1,32),"kv_heads":(1,4)}
    if any(not integer(p[k],*limit) for k,limit in limits.items()):raise ValueError("Bounded integer topology required")
    if p["chiplets"] and p["cus"]%p["chiplets"]:raise ValueError("Original topology needs equal CU counts per chiplet")
    if p["explicit_splits"] is not None and not integer(p["explicit_splits"],1,4):raise ValueError("Null or1..4split override required")
    if strategy not in ("compare","request_only"):raise ValueError("Known candidate strategy required")
    base={"status":"outside_tuned_profile","selected_splits":None,"placement":None,"valid_rows_per_chiplet":None,
          "grid_rows":None,"launched_workgroups":None,"valid_workgroups":None,"padding_workgroups":None,
          "max_valid_workgroups_per_chiplet":None,"one_round":None,"native_followup":None,
          "native_gpu_verified":None,"native_workspace_bytes":None}
    if p["requests"]==0:return {**base,"status":"no_work","grid_rows":0,"launched_workgroups":0,"valid_workgroups":0,"padding_workgroups":0}
    if p["chiplets"]<=1 or p["topk"]!=16 or p["query_rows"]!=8 or p["kv_heads"]!=1:return base
    capacity=p["cus"]//p["chiplets"];candidates=[]
    placements=("request","balanced") if strategy=="compare" else ("request",)
    for kind in placements:
        counts,grid=ownership(p["requests"],p["query_rows"],p["chiplets"],kind)
        for splits in range(1,5):
            if max(counts)*splits<=capacity:candidates.append((splits,kind,counts,grid))
    if p["explicit_splits"] is not None:
        counts,grid=ownership(p["requests"],p["query_rows"],p["chiplets"],"request")
        selected=(p["explicit_splits"],"request",counts,grid)
    elif candidates:selected=max(candidates,key=lambda c:(c[0],c[1]=="request"))
    else:
        return {**base,"status":"outside_one_round","native_followup":"occupancy_2_candidate" if p["requests"]<=48 else "plain_cost_model"}
    splits,kind,counts,grid=selected;valid=p["requests"]*p["query_rows"]*splits
    return {**base,"status":"explicit_override" if p["explicit_splits"] is not None else "one_round_candidate",
            "selected_splits":splits,"placement":kind,"valid_rows_per_chiplet":counts,"grid_rows":grid,
            "launched_workgroups":grid*splits,"valid_workgroups":valid,"padding_workgroups":grid*splits-valid,
            "max_valid_workgroups_per_chiplet":max(counts)*splits,"one_round":max(counts)*splits<=capacity}


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        req=n//8
        if ownership(req,8,8,"balanced")!=enumerate_ownership(req,8,8,"balanced")[:2]:raise AssertionError("Ownership parity")
        samples={True:[],False:[]}
        for mode in (True,False):
            for _ in range(2):
                if mode:ownership(req,8,8,"balanced")
                else:enumerate_ownership(req,8,8,"balanced")
        for repeat in range(7):
            for mode in ((True,False) if repeat%2==0 else (False,True)):
                start=time.perf_counter_ns()
                for _ in range(3):
                    if mode:ownership(req,8,8,"balanced")
                    else:enumerate_ownership(req,8,8,"balanced")
                samples[mode].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"rows":n,"analytic_samples_ms":samples[True],"enumerated_samples_ms":samples[False],
                     "analytic_median_ms":statistics.median(samples[True]),"enumerated_median_ms":statistics.median(samples[False])})
    return {"provenance":"measured","scope":"Original CPUownership tally versus grid-row enumeration,no GPU/kernel",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows,
            "warmups":2,"repeats":7,"batches_per_sample":3,"native_gpu_latency_ms":None}


def report(latency):
    variants=[("ten requests",{}),("eight requests tie",{"requests":8}),("eleven requests",{"requests":11}),
              ("thirty-three requests boundary",{"requests":33}),("two KV heads",{"kv_heads":2}),
              ("explicit4split override",{"explicit_splits":4}),("zero rows",{"requests":0})]
    return {"schema_version":1,"round":170,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"vllm-project/vllm","pin":PIN,"chronology":"actualNov14UTC09:16:29/BJT17:16:29 MiniMax M3.1 XCD balancing",
                      "license":"Apache-2.0 rights retained,no implementation mirror",
                      "verified":"request versus balanced-tail mapping,one-round split/tie policy,eligibility,occupancy2boundary and explicit overrides",
                      "scope":"original row ownership/one-round search only,not FlyDSL/NVFP4/MXFP8/GPU/complete native cost model"},
            "case":{"inputs":DEFAULT,"current":plan(DEFAULT),"request_only":plan(DEFAULT,"request_only")},
            "matrix":[{"name":name,"inputs":{**DEFAULT,**change},"outputs":plan({**DEFAULT,**change})} for name,change in variants],
            "latency":latency,"native_reported_timing_archived":False,"visual_acceptance":None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--measure",action="store_true");ap.add_argument("--check",action="store_true");a=ap.parse_args()
    path=HERE/"latency.json"
    if a.measure or not path.exists():path.write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=report(json.loads(path.read_text(encoding="utf-8")))
    from experiments.xcd_tail_placement_selection.render import outputs
    files={HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}
    for path,value in files.items():
        if a.check:
            if path.read_text(encoding="utf-8")!=value:raise AssertionError(f"Unsealed artifact:{path}")
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding="utf-8")
    print("Round170 sealed artifacts match" if a.check else "Rendered original XCDtail ownership")


if __name__=="__main__":main()
