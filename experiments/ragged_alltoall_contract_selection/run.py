"""Original single-destination token transport;no distributed backend execution."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="0a95a45a42fbec1ab15a5c8ad9b6f03ac2e225ef"
DEFAULT={"tokens":[[10,11,12],[20,21],[30],[40,41]],"destinations":[[2,0,2],[3,0],[2],[0,3]]}


def integer(v,low,high):
    if type(v) is not int or not low<=v<=high:
        raise ValueError(f"exact integer in {low}..{high} required")
    return v


def dispatch(p,declared_recv=None):
    if not isinstance(p,dict) or set(p)!={"tokens","destinations"}:
        raise ValueError("exact original tokens/destinations input required")
    tokens=p["tokens"];destinations=p["destinations"]
    if not isinstance(tokens,list) or not 1<=len(tokens)<=8 or not isinstance(destinations,list) or len(destinations)!=len(tokens):
        raise ValueError("one to eight group-relative rank rows required")
    n=len(tokens);send=[[0]*n for _ in range(n)];packed=[]
    for rank,(values,targets) in enumerate(zip(tokens,destinations)):
        if not isinstance(values,list) or len(values)>128 or not isinstance(targets,list) or len(targets)!=len(values):
            raise ValueError("matched bounded rank token/target lists required")
        for value,destination in zip(values,targets):
            integer(value,-10**6,10**6);integer(destination,0,n-1)
            send[rank][destination]+=1
        records=[{"owner":rank,"position":i,"value":v,"destination":d} for i,(v,d) in enumerate(zip(values,targets))]
        packed.append(sorted(records,key=lambda r:r["destination"]))
    recv=[[send[source][destination] for source in range(n)] for destination in range(n)]
    if declared_recv is not None:
        if not isinstance(declared_recv,list) or len(declared_recv)!=n:
            raise ValueError("one declared receive-split row per rank required")
        for destination,row in enumerate(declared_recv):
            if not isinstance(row,list) or len(row)!=n:
                raise ValueError("declared receive row must have world-size entries")
            for source,count in enumerate(row):
                integer(count,0,128)
                if count!=send[source][destination]:
                    raise ValueError(f"pairwise split mismatch source{source}→destination{destination};row totals are insufficient")
    received=[[record for source in range(n) for record in packed[source] if record["destination"]==destination] for destination in range(n)]
    responses=[[{**r,"result":2*r["value"]+100*destination} for r in records] for destination,records in enumerate(received)]
    restored=restore(responses,[len(v) for v in tokens])
    naive=[[r["result"] for records in responses for r in records if r["owner"]==owner] for owner in range(n)]
    return {"send_splits":send,"recv_splits":recv,"packed":packed,"received":received,"responses":responses,
            "restored":restored,"naive_return_order":naive,
            "naive_positions_matching":sum(a==b for actual,wrong in zip(restored,naive) for a,b in zip(actual,wrong)),
            "tokens":sum(map(len,tokens)),"send_rows":[sum(v) for v in send],"receive_rows":[sum(v) for v in recv]}


def restore(responses,lengths):
    if not isinstance(lengths,list) or not 1<=len(lengths)<=8 or not isinstance(responses,list) or len(responses)!=len(lengths):
        raise ValueError("original response rank grid and lengths required")
    for length in lengths:
        integer(length,0,128)
    restored=[[None]*n for n in lengths];seen=set()
    for destination,records in enumerate(responses):
        if not isinstance(records,list):
            raise ValueError("response rank list required")
        for record in records:
            if not isinstance(record,dict) or set(record)!={"owner","position","value","destination","result"}:
                raise ValueError("exact original response identity fields required")
            owner=integer(record["owner"],0,len(lengths)-1);position=integer(record["position"],0,127)
            integer(record["value"],-10**6,10**6);integer(record["result"],-2*10**6,2*10**6+700)
            if record["destination"]!=destination or type(record["destination"]) is not int or position>=lengths[owner] or (owner,position) in seen:
                raise ValueError("response identity duplicate,wrong destination or out of owner bounds")
            seen.add((owner,position));restored[owner][position]=record["result"]
    if len(seen)!=sum(lengths):
        raise ValueError("missing original response identity")
    return restored


def budget(p):
    if not isinstance(p,dict) or set(p)!={"world","tokens","features","element_bytes"}:
        raise ValueError("exact original selected payload fields required")
    n=integer(p["world"],1,8);tokens=integer(p["tokens"],0,1024);features=integer(p["features"],1,4096);size=integer(p["element_bytes"],1,8)
    return {"selected_one_way_token_payload_bytes":tokens*features*size,
            "selected_roundtrip_token_payload_bytes":2*tokens*features*size,
            "selected_all_rank_send_and_recv_int64_split_bytes":2*n*n*8,
            "selected_one_way_two_int64_identity_bytes":tokens*2*8,
            "scope":"selected payload only;scalar toy does not execute feature tensors;no NCCL wire,metadata exchange,allocator,workspace or peak memory"}


def timing_paths(n):
    ids=sorted(range(n),key=lambda i:(i%4,i));values=[2*i+1 for i in ids];positions={identity:i for i,identity in enumerate(ids)}
    def indexed():
        return [values[positions[i]] for i in range(n)]
    def searching():
        return [values[ids.index(i)] for i in range(n)]
    return indexed,searching


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
        rows.append({"tokens":n,"indexed_samples_ms":samples[0],"searching_samples_ms":samples[1],
                     "indexed_median_ms":statistics.median(samples[0]),"searching_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original token-ID inverse lookup with prebuilt dictionary versus repeated list.index;both output original-order values",
            "excludes":"packing/index construction,dispatch class,Pytorch,NCCL/MPI,GPUs,network,expert computation and distributed latency","rows":rows}


def build_report(latency):
    result=dispatch(DEFAULT)
    bad=[list(v) for v in result["recv_splits"]];bad[0][0],bad[0][2]=bad[0][2],bad[0][0]
    try:
        dispatch(DEFAULT,bad)
    except ValueError as error:
        mismatch=str(error)
    else:
        raise RuntimeError("original pairwise mismatch was not rejected")
    reversed_responses=[list(reversed(v)) for v in result["responses"]]
    return {"schema_version":1,"round":144,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"pytorch/pytorch","pin":PIN,"license":"PyTorch BSD-style source rights preserved;no implementation mirrored",
                      "verified":"all_to_all_single dim0split units,default equal divisibility,source-concatenated output,local dtype check,experimental API and backend-specific config",
                      "chronology":"Nov14nccl2PSK capability change in pinned tree;not a new ragged-split algorithm;no encryption/transport tested",
                      "scope":"original single-destination scalar token permutation;zero receives permitted by original policy,not native backend certification"},
            "case":{"inputs":DEFAULT,"outputs":result,"bad_receive_row":bad[0],"bad_receive_row_total":sum(bad[0]),
                    "correct_receive_row_total":sum(result["recv_splits"][0]),"mismatch_error":mismatch,
                    "reversed_processing_restored":restore(reversed_responses,[len(v) for v in DEFAULT["tokens"]]),
                    "selected_budget":{"inputs":{"world":4,"tokens":8,"features":4,"element_bytes":2},
                                       "outputs":budget({"world":4,"tokens":8,"features":4,"element_bytes":2})}},
            "latency":latency,"native_collective_latency_ms":None,"native_backend_zero_receive_verified":None,
            "native_wire_bytes":None,"gpu_peak_memory_bytes":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round144 sealed artifacts match" if a.check else "Wrote Round144 original artifacts")


if __name__=="__main__":
    main()
