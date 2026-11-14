"""Original retained-support and routing ledger;not native SGLang sampling."""

import argparse
import heapq
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="4a909c3ba6ac32ea59347f9f5d7680794557cb97"
ALL=1<<30
DEFAULT={"top_ks":[ALL,20],"keep":[1],"vocab":2048,"seeded":False,"capture":False}
CASES=[("Drop greedy,retain whole vocabulary",[1,ALL],1),
       ("Drop whole vocabulary,retain20",[ALL,20],1),
       ("Drop greedy and20,retain50",[1,20,50],2),
       ("Drop1300,retain20",[1300,20],1),
       ("Drop whole vocabulary,retain greedy",[ALL,1],1),
       ("Drop whole vocabulary,retain1024",[1024,ALL],0)]


def flags(ks):
    whole=ALL in ks
    return {"is_all_greedy":all(k==1 for k in ks),"is_any_greedy":1 in ks,
            "need_top_k_sampling":any(k!=ALL for k in ks),
            "npu_top_k_top_p_eligible":all(k<=1024 for k in ks),
            "max_top_k":max(ks) if ks and not whole else None}


def weights(n):
    return [(1777*i+19)%n+1 for i in range(n)]


def full_prefix(scores,k):
    return sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:k]


def bounded_prefix(scores,k):
    return heapq.nsmallest(k,range(len(scores)),key=lambda i:(-scores[i],i))


def support(scores,k,f,seeded,capture):
    bound=f["max_top_k"]
    if (bound is not None and bound<k) or (f["is_all_greedy"] and k!=1) or (not f["need_top_k_sampling"] and k!=ALL):
        raise ValueError("unsafe original metadata:required row support cannot be skipped or truncated")
    if f["is_all_greedy"]:
        route="greedy_argmax";width=0
        ids=[min(range(len(scores)),key=lambda i:(-scores[i],i))]
    elif not f["need_top_k_sampling"]:
        route="whole_vocab_direct";width=0;ids=list(range(len(scores)))
    elif bound is not None and bound<len(scores) and not seeded and not capture:
        route="bounded_prefix";width=bound;ids=bounded_prefix(scores,bound)[:min(k,len(scores))]
    else:
        route="full_sort";width=len(scores);ids=full_prefix(scores,len(scores))[:min(k,len(scores))]
    return {"route":route,"planned_ranked_prefix_columns":width,"support_token_ids":sorted(ids)}


def evaluate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original top_ks,keep,vocab,seeded,capture fields required")
    ks,keep,n=p["top_ks"],p["keep"],p["vocab"]
    if not isinstance(ks,list) or len(ks)>16 or any(type(k) is not int or not 1<=k<=ALL for k in ks):
        raise ValueError("bounded normalized positive top_k required;API-1must normalize to2^30 first")
    if type(n) is not int or not 2<=n<=4096:
        raise ValueError("original vocabulary2..4096 required")
    if not isinstance(keep,list) or any(type(i) is not int or not 0<=i<len(ks) for i in keep) or keep!=sorted(set(keep)):
        raise ValueError("unique increasing in-range keep indices required")
    if type(p["seeded"]) is not bool or type(p["capture"]) is not bool:
        raise ValueError("explicit original seeded/capture booleans required")
    before=flags(ks);retained=[ks[i] for i in keep]
    after=flags(retained) if 0<len(retained)<len(ks) else before
    fresh=flags(retained) if retained else None
    scores=weights(n);rows=[]
    for source,k in zip(keep,retained):
        old=support(scores,k,before,p["seeded"],p["capture"])
        new=support(scores,k,after,p["seeded"],p["capture"])
        rows.append({"source_row":source,"top_k":k,"sticky":old,"refreshed":new,
                     "same_original_support":old["support_token_ids"]==new["support_token_ids"]})
    return {"before":before,"after":after,"fresh_nonempty_flags":fresh,
            "matches_fresh_nonempty_flags":after==fresh if fresh is not None else None,
            "status":"retained" if retained else "empty_no_forward",
            "retained_rows":len(rows),"support_misses":sum(not r["same_original_support"] for r in rows),
            "rows":rows,"native_sampled_tokens":None,"native_npu_execution":None}


def measure():
    rows=[];k=20
    for n in (64,256,1024,4096):
        scores=weights(n)
        funcs=(lambda:bounded_prefix(scores,k),lambda:full_prefix(scores,k))
        if funcs[0]()!=funcs[1]():
            raise RuntimeError("original bounded/full top20 support order differs")
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
        rows.append({"vocab":n,"k":k,"bounded_samples_ms":samples[0],"full_samples_ms":samples[1],
                     "bounded_median_ms":statistics.median(samples[0]),"full_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original CPUheapq.nsmallest versus sorted for identical top20ranked token IDs",
            "excludes":"score/input setup,flag refresh,softmax/RNG/SGLang/Torch/GPU/NPU/latency or throughput of native sampler",
            "rows":rows}


def build_report(latency):
    matrix=[{"name":name,"inputs":{**DEFAULT,"top_ks":ks,"keep":[i]},
             "outputs":evaluate({**DEFAULT,"top_ks":ks,"keep":[i]})} for name,ks,i in CASES]
    rows=[r for c in matrix for r in c["outputs"]["rows"]]
    robustness=[]
    for changes in ({"seeded":True},{"capture":True},{"top_ks":[20,ALL],"keep":[0]},{"keep":[]}):
        p={**DEFAULT,**changes};robustness.append({"inputs":p,"outputs":evaluate(p)})
    return {"schema_version":1,"round":154,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"sgl-project/sglang","pin":PIN,"license":"Apache-2.0 rights retained,no implementation mirror",
                      "chronology":"actualNov14UTC05:21:40/BJT13:21:40running-batch top-k flag refresh",
                      "verified":"safe-but-sticky SamplingBatchInfoflags;ScheduleBatchrefreshes from kept request params;normalizedTOP_K_ALL2^30",
                      "scope":"original canonical token-support and selected-plan prefix ledger,not native sampled tokens"},
            "case":{"inputs":DEFAULT,"outputs":evaluate(DEFAULT)},"matrix":matrix,"robustness":robustness,
            "summary":{"retained_rows":len(rows),"support_misses":sum(not r["same_original_support"] for r in rows),
                       "sticky_prefix_columns":sum(r["sticky"]["planned_ranked_prefix_columns"] for r in rows),
                       "refreshed_prefix_columns":sum(r["refreshed"]["planned_ranked_prefix_columns"] for r in rows),
                       "tightened_plans":sum(r["sticky"]["planned_ranked_prefix_columns"]>r["refreshed"]["planned_ranked_prefix_columns"] for r in rows)},
            "latency":latency,"native_sglang_verified":None,"native_seeded_tie_order":None,"native_gpu_latency_ms":None}


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
    print("Round154sealed artifacts match" if args.check else "Wrote Round154original artifacts")


if __name__=="__main__":
    main()
