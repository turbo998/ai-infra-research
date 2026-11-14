"""Original communicator-membership and cyclic-row ownership ledger;no NCCL execution."""

import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="0a95a45a42fbec1ab15a5c8ad9b6f03ac2e225ef"
DEFAULT={"parent_global_ranks":[7,2,19,5,11,17],"excluded":[1,4],"callers":[0,2,3,5],
         "mode":"default","outstanding":0,"shrink_share":True,"rows":list(range(100,112)),
         "checkpoint":[[1,101],[4,104],[7,107],[10,110]],"features":3,"element_bytes":2}


def integer(value,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError(f"exact integer in {low}..{high} required")
    return value


def validate(p):
    fields={"parent_global_ranks","excluded","callers","mode","outstanding","shrink_share","rows","checkpoint","features","element_bytes"}
    if not isinstance(p,dict) or set(p)!=fields:
        raise ValueError("exact original membership/row-ledger fields required")
    globals_=p["parent_global_ranks"]
    if not isinstance(globals_,list) or not 2<=len(globals_)<=16:
        raise ValueError("two to sixteen original parent-group entries required")
    for value in globals_:
        integer(value,0,10**6)
    if len(set(globals_))!=len(globals_):
        raise ValueError("unique synthetic global rank labels required")
    world=len(globals_)
    for name in ("excluded","callers"):
        if not isinstance(p[name],list):
            raise ValueError("parent-relative rank lists required")
        for rank in p[name]:
            integer(rank,0,world-1)
        if len(set(p[name]))!=len(p[name]):
            raise ValueError("duplicate parent-relative rank rejected")
    if not 1<=len(p["excluded"])<world:
        raise ValueError("original PyTorch-boundary exclude list must be nonempty and leave survivors")
    if p["mode"] not in ("default","abort") or type(p["shrink_share"]) is not bool:
        raise ValueError("original default/abort mode and exact share boolean required")
    integer(p["outstanding"],0,64);integer(p["features"],1,4096);integer(p["element_bytes"],1,8)
    if not isinstance(p["rows"],list) or len(p["rows"])>128:
        raise ValueError("at most128original scalar row references required")
    for value in p["rows"]:
        integer(value,-10**6,10**6)
    if not isinstance(p["checkpoint"],list) or len(p["checkpoint"])>len(p["rows"]):
        raise ValueError("bounded original checkpoint record list required")
    seen=set()
    for record in p["checkpoint"]:
        if not isinstance(record,list) or len(record)!=2:
            raise ValueError("checkpoint row-ID/value pairs required")
        identity=integer(record[0],0,len(p["rows"])-1);integer(record[1],-10**6,10**6)
        if identity in seen:
            raise ValueError("duplicate checkpoint row identity")
        if record[1]!=p["rows"][identity]:
            raise ValueError("checkpoint value disagrees with original synthetic reference")
        seen.add(identity)


def membership(p):
    validate(p)
    world=len(p["parent_global_ranks"]);excluded=set(p["excluded"])
    survivors=[rank for rank in range(world) if rank not in excluded]
    if set(p["callers"])!=set(survivors):
        raise ValueError("callers must be exactly surviving parent-relative ranks;excluded ranks must not call")
    if p["mode"]=="default" and p["outstanding"]:
        raise ValueError("default shrink requires no outstanding work;drain or select explicit abort semantics")
    members=[{"parent_rank":rank,"new_rank":rank-sum(x<rank for x in excluded),"global_id":p["parent_global_ranks"][rank]} for rank in survivors]
    return {"parent_world":world,"new_world":len(survivors),"members":members,"excluded":sorted(excluded),
            "changed_rank_ids":sum(v["parent_rank"]!=v["new_rank"] for v in members),
            "abort_requested":p["mode"]=="abort",
            "selected_sharing_request_permitted":p["mode"]=="default" and p["shrink_share"],
            "scope":"original preflight and association ledger;not observed communicators,abort completion or actual resource sharing"}


def simulate(p):
    group=membership(p);world=group["parent_world"];new_world=group["new_world"]
    remap={v["parent_rank"]:v["new_rank"] for v in group["members"]};excluded=set(group["excluded"])
    retained=[{"row":i,"value":v,"parent_owner":i%world,"new_resident":remap[i%world],"target_owner":i%new_world}
              for i,v in enumerate(p["rows"]) if i%world not in excluded]
    lost=[i for i in range(len(p["rows"])) if i%world in excluded]
    checkpoint=dict(p["checkpoint"]);missing=[i for i in lost if i not in checkpoint]
    moved=[v["row"] for v in retained if v["new_resident"]!=v["target_owner"]]
    row_bytes=p["features"]*p["element_bytes"]
    selected={"retained_rows":len(retained),"lost_rows":len(lost),"already_at_target_rows":len(retained)-len(moved),
              "survivor_rows_needing_move":len(moved),"selected_total_feature_bytes":len(p["rows"])*row_bytes,
              "selected_survivor_move_feature_bytes":len(moved)*row_bytes,"selected_checkpoint_restore_feature_bytes":len(lost)*row_bytes,
              "scope":"selected feature payload only;scalar ledger does not execute feature tensors or model NCCL wire,checkpoint staging or allocator peak"}
    error="missing checkpoint row IDs "+json.dumps(missing,separators=(",",":")) if missing else None
    resharded=None
    if not missing:
        resharded=[[] for _ in range(new_world)]
        for identity,value in enumerate(p["rows"]):
            resharded[identity%new_world].append([identity,checkpoint[identity] if identity in lost else value])
    return {"membership":group,"retained":retained,"lost_row_ids":lost,"survivor_row_ids_needing_move":moved,
            "selected_budget":selected,"data_ready":not missing,"data_error":error,"resharded":resharded,
            "scope":"membership rank compression is separate from explicit original cyclic-row recovery and redistribution;not training recovery"}


def probe(name,**changes):
    p={**copy.deepcopy(DEFAULT),**changes}
    try:
        result=simulate(p)
    except ValueError as error:
        return {"name":name,"inputs":p,"outputs":None,"preflight_error":str(error)}
    return {"name":name,"inputs":p,"outputs":result,"preflight_error":None}


def cases():
    return [probe("complete_original_recovery"),probe("missing_checkpoint",checkpoint=[]),
            probe("excluded_rank_calls",callers=[0,1,2,3,5]),probe("survivor_absent",callers=[0,2,3]),
            probe("default_has_pending_work",outstanding=2),probe("abort_with_pending_metadata",outstanding=2,mode="abort"),
            probe("reversed_exclusion_and_call_order",excluded=[4,1],callers=[5,3,2,0])]


def timing_paths(n):
    excluded=list(range(1,n,4));ex=set(excluded);survivors=[rank for rank in range(n) if rank not in ex]
    removed=0;prefix=[]
    for rank in range(n):
        prefix.append(rank-removed)
        removed+=rank in ex
    def indexed():
        return [(rank,prefix[rank]) for rank in survivors]
    def counted():
        return [(rank,rank-sum(x<rank for x in excluded)) for rank in survivors]
    return indexed,counted


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=timing_paths(n);assert funcs[0]()==funcs[1]()
        for _ in range(2):
            for f in funcs:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    funcs[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"parent_rank_entries":n,"prefix_samples_ms":samples[0],"counted_samples_ms":samples[1],
                     "prefix_median_ms":statistics.median(samples[0]),"counted_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original global association planner:prebuilt exclusion-prefix lookup versus repeated less-than counts for every survivor;one in four excluded",
            "excludes":"prefix/exclusion/survivor construction,native per-rank C++,NCCL collectives,GPUs,network,state recovery and redistribution;standalone planner sizes exceed UI bound","rows":rows}


def build_report(latency):
    return {"schema_version":1,"round":146,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"pytorch/pytorch","pin":PIN,"license":"PyTorch BSD-style source rights preserved;NVIDIA API facts paraphrased,no implementation mirrored",
                      "verified":"survivors call shrink;excluded must not call;nonempty/non-all/unique parent-relative excludes;new rank=old minus lower excluded;default pending-work check;NVIDIA default/abort/sharing semantics",
                      "chronology":"Nov14shrink child-option/cleanup integration change;not invention of NCCL shrink or automatic state recovery",
                      "scope":"original global association ledger and cyclic-row distribution;checkpoint contents validated against original synthetic reference,not a live tensor checksum"},
            "case":{"inputs":DEFAULT,"outputs":simulate(DEFAULT)},"scenarios":cases(),"latency":latency,
            "native_shrink_latency_ms":None,"native_abort_completed":None,"native_resource_sharing_verified":None,
            "native_state_recovery_verified":None,"native_wire_bytes":None,"gpu_peak_memory_bytes":None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if args.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round146 sealed artifacts match" if args.check else "Wrote Round146 original artifacts")


if __name__=="__main__":
    main()
