"""Original joint-victim/reclamation ledger, not native trie or shared allocator."""

import argparse
import copy
import heapq
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
PIN="1dd0dcce38ee01c6828c16700e7a4790bf438a9d"
NODE_FIELDS={"nid","parent","full","swa","full_tick","swa_tick","swa_rank",
             "full_units","swa_units","full_locked","swa_locked"}


def node(nid,parent,ft,fu,st=None,su=0,rank=0):
    return {"nid":nid,"parent":parent,"full":True,"swa":st is not None,"full_tick":ft,
            "swa_tick":st,"swa_rank":rank,"full_units":fu,"swa_units":su,
            "full_locked":False,"swa_locked":False}


DEFAULT={"policy":"joint_lru","tree_enabled":True,"windows":1,"state_slot":False,
         "chain_allocator":True,"chain_window":True,"chain_mamba":False,
         "core_supports_joint":True,"session_cache":False,"target_units":8,
         "nodes":[node(0,None,2,16),node(1,0,9,32,1,8,0),
                  node(2,None,5,16),node(3,None,8,16,3,8,1)]}


def integer(v,maxval=2**30):
    return type(v) is int and 0<=v<=maxval


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original feature/forest fields required")
    if p["policy"] not in ("lru","joint_lru") or not integer(p["windows"],4) or not integer(p["target_units"]):
        raise ValueError("known policy and bounded window/target required")
    for k in ("tree_enabled","state_slot","chain_allocator","chain_window","chain_mamba","core_supports_joint","session_cache"):
        if type(p[k]) is not bool:raise ValueError("explicit feature witnesses required")
    rows=p["nodes"]
    if not isinstance(rows,list) or len(rows)>64:raise ValueError("bounded original forest required")
    for r in rows:
        if not isinstance(r,dict) or set(r)!=NODE_FIELDS:raise ValueError("exact original node fields required")
        if not integer(r["nid"],4095) or r["parent"] is not None and not integer(r["parent"],4095):
            raise ValueError("bounded node/parent identity required")
        if any(type(r[k]) is not bool for k in ("full","swa","full_locked","swa_locked")):
            raise ValueError("explicit component/lock witnesses required")
        if any(not integer(r[k]) for k in ("full_tick","swa_rank","full_units","swa_units")):
            raise ValueError("bounded ticks/ranks/resource units required")
        if r["swa"] and not integer(r["swa_tick"]) or not r["swa"] and r["swa_tick"] is not None:
            raise ValueError("live SWA has tick;absent SWA has null tick")
        if r["full"] and not r["full_units"] or r["swa"] and not r["swa_units"]:
            raise ValueError("live components require positive declared units")
    ids={r["nid"] for r in rows}
    if len(ids)!=len(rows):raise ValueError("unique node identities required")
    live_ranks=[r["swa_rank"] for r in rows if r["swa"]]
    if len(set(live_ranks))!=len(live_ranks):raise ValueError("unique original SWA LRU ranks required")
    byid={r["nid"]:r for r in rows}
    for r in rows:
        if r["parent"] is not None and r["parent"] not in ids:raise ValueError("parent absent from original forest")
        seen=set();current=r["nid"]
        while current is not None:
            if current in seen:raise ValueError("original forest cannot contain cycles")
            seen.add(current);current=byid[current]["parent"]


def feature_mode(p):
    active=(p["policy"]=="joint_lru" and p["tree_enabled"] and p["windows"]==1 and not p["state_slot"]
            and p["chain_allocator"] and p["chain_window"] and not p["chain_mamba"])
    if active and (p["session_cache"] or not p["core_supports_joint"]):return "reject_unsupported_joint"
    return "joint" if active else "component_lru"


def split_recency(full_tick,swa_tick,prefix_access,joint=True):
    return {"suffix_full_tick":full_tick if joint else prefix_access,
            "prefix_swa_tick":swa_tick,"suffix_swa_tick":swa_tick}


def plan(p,collapse_swa_proposal=False,linear_reference=False):
    validate(p)
    mode=feature_mode(p)
    result={"mode":mode,"trace":[],"freed_units":0,"retained_full_units":sum(r["full_units"] for r in p["nodes"] if r["full"]),
            "target_met":None,"stale_entries_skipped":0,"states":[],
            "native_rust_verified":None,"native_allocator_verified":None}
    if mode!="joint":return result
    rows=copy.deepcopy(p["nodes"]);byid={r["nid"]:r for r in rows}
    def full_eligible(r):
        return (r["full"] and not r["full_locked"] and not r["swa_locked"]
                and not any(c["parent"]==r["nid"] and c["full"] for c in rows))
    def eligible(r,ct):return full_eligible(r) if ct==0 else r["swa"] and not r["swa_locked"]
    def entry(r,ct):return (r["full_tick"] if ct==0 else r["swa_tick"],ct,r["nid"] if ct==0 else r["swa_rank"],r["nid"])
    heap=[entry(r,ct) for r in rows for ct in (0,1) if eligible(r,ct)]
    heapq.heapify(heap)
    while result["freed_units"]<p["target_units"]:
        if linear_reference:
            candidates=[entry(r,ct) for r in rows for ct in (0,1) if eligible(r,ct)]
            if not candidates:break
            _,ct,_,nid=min(candidates)
        else:
            while heap:
                _,ct,_,nid=heapq.heappop(heap)
                if eligible(byid[nid],ct):break
                result["stale_entries_skipped"]+=1
            else:break
        r=byid[nid];freed=0
        if ct==1:
            freed=r["swa_units"];r["swa"]=False;r["swa_tick"]=None
            if collapse_swa_proposal and r["full"]:
                freed+=r["full_units"];r["full"]=False
        else:
            freed=r["full_units"];r["full"]=False
            if r["swa"]:
                freed+=r["swa_units"];r["swa"]=False;r["swa_tick"]=None
        result["freed_units"]+=freed
        result["trace"].append({"component":"full" if ct==0 else "swa","nid":nid,"freed_units":freed,
                                "full_retained":r["full"],"swa_tick_after":r["swa_tick"]})
        parent=byid.get(r["parent"])
        if not linear_reference and parent is not None and full_eligible(parent):
            heapq.heappush(heap,entry(parent,0))
    result["retained_full_units"]=sum(r["full_units"] for r in rows if r["full"])
    result["target_met"]=result["freed_units"]>=p["target_units"]
    result["states"]=[{"nid":r["nid"],"full":r["full"],"swa":r["swa"],"swa_tick":r["swa_tick"]} for r in rows]
    return result


def measure():
    data=[]
    for size in (64,256,1024,4096):
        entries=[((i*37)%size,i%2,i,i) for i in range(size)]
        def heap_queue():
            h=entries.copy();heapq.heapify(h);out=[]
            while h:out.append(heapq.heappop(h))
            return out
        def min_queue():
            h=entries.copy();out=[]
            while h:
                v=min(h);h.remove(v);out.append(v)
            return out
        funcs=(heap_queue,min_queue)
        if heap_queue()!=min_queue():raise RuntimeError("original victim queue order differs")
        for _ in range(2):
            for f in funcs:f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                start=time.perf_counter_ns()
                for _ in range(3):funcs[i]()
                samples[i].append((time.perf_counter_ns()-start)/3/1e6)
        data.append({"candidates":size,"heap_samples_ms":samples[0],"linear_samples_ms":samples[1],
                     "heap_median_ms":statistics.median(samples[0]),"linear_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original static victim-key queue drain,heap versus repeated linear min",
            "excludes":"setup,dynamic tree operations,native Rust/Torch/bindings,allocator compaction,backup barriers,GPUs,cache hit rate","rows":data}


def build_report(latency):
    variants=[("swa_strip_only",DEFAULT),("parent_reentry",{**DEFAULT,"target_units":1000}),
              ("full_pin",{**DEFAULT,"nodes":[{**r,"full_locked":True} if r["nid"]==1 else r for r in DEFAULT["nodes"]]}),
              ("inactive_multiple_windows",{**DEFAULT,"windows":2}),
              ("unsupported_core",{**DEFAULT,"core_supports_joint":False}),
              ("full_cascade_stale_swa",{**DEFAULT,"target_units":1000,"nodes":[{**r,"full_tick":0,"swa_tick":20} if r["nid"]==1 else r for r in DEFAULT["nodes"]]})]
    return {"schema_version":1,"round":163,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"sgl-project/sglang","pin":PIN,"chronology":"actualNov14UTC01:53:13/BJT09:53:13 Rust joint Full/SWA eviction",
                      "license":"Apache-2.0 rights retained,no implementation mirror",
                      "verified":"uses_joint_lru gate;Full/SWA shared arena ticks;minheap ties;SWA strip-only including pending finish;Full parent reentry;stale candidates;split recency and SWA tick cleanup",
                      "scope":"original victim/resource-unit forest,not native radix tree,allocator compaction or async backup protocol"},
            "case":{"inputs":DEFAULT,"current":plan(DEFAULT),"collapse_proposal":plan(DEFAULT,True),
                    "baseline_label":"counterfactual SWA-to-Full collapse,not old Rust execution;active joint mode previously unsupported"},
            "matrix":[{"name":name,"inputs":p,"outputs":plan(p)} for name,p in variants],
            "split_case":{"inputs":{"full_tick":5,"swa_tick":3,"prefix_access":20},"joint":split_recency(5,3,20),"other_policy":split_recency(5,3,20,False)},
            "latency":latency,"native_rust_verified":None,"native_allocator_verified":None,"browser_visual_acceptance":None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--measure",action="store_true");ap.add_argument("--check",action="store_true");a=ap.parse_args()
    if a.measure:(HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text()))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:raise SystemExit(f"stale artifact:{file}")
        else:file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round163 sealed artifacts match" if a.check else "Wrote Round163 original artifacts")


if __name__=="__main__":main()
