"""Original builder/layout separation;not vLLM,Torch or CUDA graph execution."""

import argparse
import json
import math
from pathlib import Path
import platform
import re
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="165156b59ba013a6353b70237e11366900cb99d3"
CG={"NEVER":0,"UNIFORM_BATCH":1,"ALWAYS":2}
DEFAULT={"manager_block":256,"allocated_kernel":64,"blocks":4,"max_len":1024,"max_seqs":2,"ubatches":2,
         "no_metadata_layers":["draft"],"address":{"block":3,"position":250},
         "groups":[{"name":"target","layers":["target"],"fixed":[],"multiples":[16],"cg":"ALWAYS","owns_workspace":False,"workspace_bytes":0},
                   {"name":"draft","layers":["draft"],"fixed":[16,32,64],"multiples":[],"cg":"UNIFORM_BATCH","owns_workspace":True,"workspace_bytes":4096}]}


def bounded(v,lo,hi,name):
    if type(v) is not int or not lo<=v<=hi:
        raise ValueError(f"bounded original{name}integer required")


def name(v):
    if not isinstance(v,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}",v):
        raise ValueError("bounded original ASCIIgroup/layer name required")


def supports(g,k):
    return k in g["fixed"] or any(k%m==0 for m in g["multiples"])


def choose(manager,groups):
    candidates=[k for k in range(1,manager+1) if manager%k==0 and all(supports(g,k) for g in groups)]
    if not candidates:
        raise ValueError("no common original divisible kernel block across all layout groups")
    return max(candidates)


def plan(p,k,skip):
    width=math.ceil(p["max_len"]/k);stride=p["manager_block"]//k;rows=[]
    owner=None;workspace=0
    for g in p["groups"]:
        count=0 if set(g["layers"]).issubset(skip) else p["ubatches"]
        if count and owner is None and g["owns_workspace"]:
            owner=g["name"];workspace=g["workspace_bytes"]
        rows.append({"group":g["name"],"layers":g["layers"],"builders":count,
                     "kernel_block":k,"offset_entries":stride if stride>1 else 0,
                     "original_unpadded_table_bytes":count*p["max_seqs"]*width*4 if stride>1 else 0,
                     "original_offset_bytes":stride*4 if stride>1 else 0,"cg":g["cg"]})
    active=[g for g in rows if g["builders"]]
    return {"groups":rows,"builder_count":sum(g["builders"] for g in rows),
            "metadata_layer_names":[layer for g in active for layer in g["layers"]],
            "attention_metadata_cg_support":min((g["cg"] for g in active),key=lambda c:CG[c],default="ALWAYS"),
            "selected_workspace_owner":owner,"selected_workspace_bytes":workspace,
            "original_unpadded_table_bytes":sum(g["original_unpadded_table_bytes"] for g in rows),
            "original_offset_bytes":sum(g["original_offset_bytes"] for g in rows)}


def evaluate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original geometry/groups/skip/address fields required")
    for field,lo,hi in (("manager_block",1,1024),("blocks",1,64),("max_len",1,65536),("max_seqs",1,64),("ubatches",1,4)):
        bounded(p[field],lo,hi,field)
    groups=p["groups"]
    if not isinstance(groups,list) or not 1<=len(groups)<=16:
        raise ValueError("original1..16layout groups required")
    seen=set();layers=set()
    for g in groups:
        if not isinstance(g,dict) or set(g)!=set(DEFAULT["groups"][0]):
            raise ValueError("exact original group fields required")
        name(g["name"])
        if g["name"] in seen:
            raise ValueError("unique original group names required")
        seen.add(g["name"])
        if not isinstance(g["layers"],list) or not 1<=len(g["layers"])<=4:
            raise ValueError("original1..4unique layers per group required")
        for layer in g["layers"]:
            name(layer)
            if layer in layers:
                raise ValueError("layer must belong to exactly one original group")
            layers.add(layer)
        for field in ("fixed","multiples"):
            if not isinstance(g[field],list) or len(g[field])>16:
                raise ValueError("bounded original support declarations required")
            for v in g[field]:
                bounded(v,1,1024,field)
        if not g["fixed"] and not g["multiples"]:
            raise ValueError("explicit original kernel support required")
        if not isinstance(g["cg"],str) or g["cg"] not in CG or type(g["owns_workspace"]) is not bool:
            raise ValueError("original attention-only CG/owner metadata required")
        bounded(g["workspace_bytes"],0,1048576,"workspace_bytes")
    skip=p["no_metadata_layers"]
    if not isinstance(skip,list) or any(not isinstance(v,str) or v not in layers for v in skip) or len(set(skip))!=len(skip):
        raise ValueError("unique known no-metadata layer names required")
    common=choose(p["manager_block"],groups)
    k=p["allocated_kernel"]
    if k is None:
        k=common
    else:
        bounded(k,1,p["manager_block"],"allocated_kernel")
        if p["manager_block"]%k or not all(supports(g,k) for g in groups):
            raise ValueError("provided allocated kernel must remain valid for every layout group")
    address=p["address"]
    if not isinstance(address,dict) or set(address)!={"block","position"}:
        raise ValueError("exact original block/position address required")
    bounded(address["block"],0,p["blocks"]-1,"block");bounded(address["position"],0,p["manager_block"]-1,"position")
    stride=p["manager_block"]//k
    mapped={"kernel_block_id":address["block"]*stride+address["position"]//k,"position":address["position"]%k}
    active_layout=[g for g in groups if not set(g["layers"]).issubset(skip)]
    naive=choose(p["manager_block"],active_layout)
    before=plan(p,k,set());after=plan(p,k,set(skip))
    return {"largest_common_kernel":common,"allocated_kernel_used":k,"stride":stride,
            "naive_remove_metadata_free_groups_kernel":naive,
            "naive_kernel_valid_for_all_layout_groups":all(supports(g,naive) for g in groups),
            "same_linear_token_address":address["block"]*p["manager_block"]+address["position"]==mapped["kernel_block_id"]*k+mapped["position"],
            "mapped_address":mapped,"before":before,"after":after,
            "original_kv_payload_bytes":p["blocks"]*p["manager_block"]*len(layers)*1*64*2*2,
            "geometry":"original unpadded per-layer KV;1head,64dimensions,2byte dtype,K+V;not packed/native allocator",
            "native_global_cudagraph_readiness":None,"native_workspace_peak_bytes":None}


def construction_paths(n):
    selected=[i for i in range(n) if i%4==0]
    def make(i):
        return {"group":i,"state":bytearray(256)}
    def gated():
        return [make(i) for i in selected]
    def eager():
        all_objects=[make(i) for i in range(n)]
        return [all_objects[i] for i in selected]
    return gated,eager


def measure():
    rows=[]
    for n in (16,64,256,1024):
        funcs=construction_paths(n)
        if funcs[0]()!=funcs[1]():
            raise RuntimeError("original gated/eager active builder objects differ")
        for _ in range(2):
            for f in funcs:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                start=time.perf_counter_ns()
                for _ in range(3):
                    funcs[i]()
                samples[i].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"candidate_objects":n,"active_objects":n//4,"private_state_bytes_each":256,
                     "gated_samples_ms":samples[0],"eager_samples_ms":samples[1],
                     "gated_median_ms":statistics.median(samples[0]),"eager_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original dict constructors with private256Bzero-state,gated versus eager then select,25percent retained",
            "excludes":"selection/input setup,shared native workspace,KVallocation,vLLM/Torch/CUDAgraphs/GPU;private dummy state is not native workspace","rows":rows}


def build_report(latency):
    mixed={**DEFAULT,"groups":[{**DEFAULT["groups"][1],"name":"mixed","layers":["target","draft"]}]}
    zero={**DEFAULT,"no_metadata_layers":[],"groups":[DEFAULT["groups"][0],{**DEFAULT["groups"][1],"workspace_bytes":0}]}
    smaller={**DEFAULT,"allocated_kernel":32,"groups":[{**DEFAULT["groups"][0],"fixed":[32],"multiples":[]},DEFAULT["groups"][1]]}
    variants=[("Context-only draft",DEFAULT),("Ordinary draft",{**DEFAULT,"no_metadata_layers":[]}),
              ("Mixed active/skipped group",mixed),("All metadata groups skipped",{**DEFAULT,"no_metadata_layers":["target","draft"]})]
    return {"schema_version":1,"round":155,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"vllm-project/vllm","pin":PIN,"license":"Apache-2.0 rights retained,no implementation mirror",
                      "chronology":"actualNov14UTC06:16:51/BJT14:16:51 context-KV-only draft metadata correction",
                      "verified":"zero builders only for fully skipped groups;kernel layout preserved;metadata workspace/CG/build traversal filtered;provided target kernel reused",
                      "scope":"original plain per-layer kernel/metadata/selected-byte ledger;not native allocation or global graph readiness"},
            "case":{"inputs":DEFAULT,"outputs":evaluate(DEFAULT)},
            "matrix":[{"name":name,"inputs":p,"outputs":evaluate(p)} for name,p in variants],
            "robustness":[{"name":"Zero-byte owned workspace","inputs":zero,"outputs":evaluate(zero)},
                          {"name":"Provided32not draft-only64","inputs":smaller,"outputs":evaluate(smaller),
                           "drafter_only_recomputed_kernel":choose(smaller["manager_block"],[smaller["groups"][1]])}],
            "latency":latency,"native_vllm_verified":None,"native_graph_capture_verified":None,"native_kv_quality":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");args=p.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text()))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if args.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round155sealed artifacts match" if args.check else "Wrote Round155original artifacts")


if __name__=="__main__":
    main()
