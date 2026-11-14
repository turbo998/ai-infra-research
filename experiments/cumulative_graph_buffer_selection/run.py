"""Original 1-D metadata/copy-policy model; no CUDA graph or FlashInfer."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
PIN="1a594a3ec1872b40ac00131a31d1f83d81cbdd80"
DEFAULT={"key":"cu_seqlens","layout":"single","src":[0,3,8],"capacity":5,
         "previous":[0,1,2,3,4],"override":None,"shared":True}
CUMULATIVE={"cu_seqlens","cu_window_seqlens"}


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original buffer fields required")
    if not isinstance(p["key"],str) or not p["key"] or p["layout"] not in ("single","packed","generic","scalar"):
        raise ValueError("explicit key and known source layout required")
    if p["override"] not in (None,"zeros","repeat","sections","prefix_only") or type(p["shared"]) is not bool:
        raise ValueError("known original override and explicit shared witness required")
    if type(p["capacity"]) is not int or not 1<=p["capacity"]<=4096:
        raise ValueError("original capacity1..4096 required")
    if not isinstance(p["previous"],list) or len(p["previous"])!=p["capacity"] or any(
            type(v) is not int or not 0<=v<=2**30 for v in p["previous"]):
        raise ValueError("complete bounded previous buffer required")
    src=p["src"]
    if src is None:
        return
    if p["layout"]=="scalar":
        if type(src) is not int or not 0<=src<=2**30 or p["capacity"]!=1:
            raise ValueError("original scalar needs one capture slot")
        return
    if not isinstance(src,list) or len(src)>4096 or any(type(v) is not int or not 0<=v<=2**30 for v in src):
        raise ValueError("bounded original integer source vector required")
    if p["layout"] in ("single","packed"):
        if not src:
            raise ValueError("cumulative metadata needs at least its zero offset")
        if p["layout"]=="packed" and (len(src)%2 or p["capacity"]%2):
            raise ValueError("packed Q/Koffsets need equal-sized sections")
        for section in split(src,p["layout"]):
            if section[0]!=0 or any(a>b for a,b in zip(section,section[1:])):
                raise ValueError("each cumulative section must start0and be nondecreasing")


def split(src,layout):
    if layout=="packed":
        return [src[:len(src)//2],src[len(src)//2:]]
    return [src]


def expected(src,capacity,layout):
    if layout=="scalar":
        return [src]
    if layout=="generic":
        return src+[0]*(capacity-len(src))
    width=capacity//2 if layout=="packed" else capacity
    return [v for section in split(src,layout) for v in section+[section[-1]]*(width-len(section))]


def copy_buffer(dst,src,rule):
    written=[False]*len(dst)
    def put(i,v):
        dst[i]=v;written[i]=True
    if rule=="sections":
        if not src or len(src)%2 or len(dst)%2:
            raise ValueError("section override needs nonempty even-sized vectors")
        width=len(dst)//2
        for start,section in zip((0,width),split(src,"packed")):
            for i in range(width):
                put(start+i,section[i] if i<len(section) else section[-1])
    else:
        if rule=="repeat" and not src:
            raise ValueError("repeat policy cannot infer an empty source terminal")
        for i in range(len(dst)):
            if i<len(src):
                put(i,src[i])
            elif rule!="prefix_only":
                put(i,src[-1] if rule=="repeat" else 0)
    return written


def evaluate(p,old=False):
    validate(p);src=p["src"];buf=p["previous"][:]
    if src is None:
        if p["shared"]:
            raise ValueError("shared capture buffer requires replay values;cannot retain another budget")
        return {"status":"captured_constant_retained","buffer":buf,"rule":None,"words_written":0,
                "full_buffer_written":False,"word_misses":None,"native_graph_verified":None}
    n=1 if p["layout"]=="scalar" else len(src)
    if n>p["capacity"]:
        return {"status":"eager_required","buffer":buf,"rule":None,"words_written":0,
                "full_buffer_written":False,"word_misses":None,"native_graph_verified":None}
    if p["layout"]=="scalar":
        rule="scalar_copy";buf[0]=src;written=[True]
    else:
        rule=p["override"] or ("repeat" if not old and p["key"] in CUMULATIVE else "zeros")
        written=copy_buffer(buf,src,rule)
    target=expected(src,p["capacity"],p["layout"])
    misses=sum(a!=b for a,b in zip(buf,target))
    full=all(written)
    return {"status":"valid_policy" if misses==0 and full else "invalid_padding_policy",
            "buffer":buf,"rule":rule,"words_written":sum(written),"full_buffer_written":full,
            "word_misses":misses,"expected_buffer":target,"logical_int32_buffer_bytes":4*len(buf),
            "native_graph_verified":None}


def paths(n):
    src=[2*i for i in range(n//4)]
    a=[99]*n;b=[99]*n
    def direct():
        for i in range(n):
            a[i]=src[i] if i<len(src) else src[-1]
        return a
    def redundant_zero():
        for i in range(n):
            b[i]=0
        for i in range(n):
            b[i]=src[i] if i<len(src) else src[-1]
        return b
    return direct,redundant_zero


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=paths(n)
        if funcs[0]()!=funcs[1]():
            raise RuntimeError("original overwrite paths differ")
        for _ in range(2):
            for f in funcs:f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                start=time.perf_counter_ns()
                for _ in range(3):funcs[i]()
                samples[i].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"words":n,"direct_samples_ms":samples[0],"redundant_samples_ms":samples[1],
                     "direct_median_ms":statistics.median(samples[0]),"redundant_median_ms":statistics.median(samples[1]),
                     "direct_writes":n,"redundant_writes":2*n})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original preallocated Pythonlist overwrite versus redundant zero-and-overwrite",
            "excludes":"setup,CUDA/Torch/H2D,graphs,FlashInfer,attention/model throughput;not native copy timings","rows":rows}


def build_report(latency):
    variants=[("single",{}),("window",{"key":"cu_window_seqlens"}),("zero_sequences",{"src":[0]}),
              ("exact_fit",{"src":[0,2,5,9,13]}),
              ("packed",{"layout":"packed","src":[0,3,8,0,4,9],"capacity":10,
                         "previous":[0,1,2,3,4,0,5,6,7,8],"override":"sections"}),
              ("packed_zero",{"layout":"packed","src":[0,0],"capacity":6,
                              "previous":[0,1,2,0,3,4],"override":"sections"}),
              ("ordinary",{"key":"position_ids","layout":"generic","src":[7,8]}),
              ("scalar",{"key":"max_seqlen","layout":"scalar","src":42,"capacity":1,
                         "previous":[9],"override":"prefix_only"})]
    matrix=[]
    for name,change in variants:
        p={**DEFAULT,**change}
        before={**p,"override":"repeat"} if p["layout"]=="packed" else p
        matrix.append({"name":name,"inputs":p,"before":evaluate(before,True),"after":evaluate(p)})
    return {"schema_version":1,"round":160,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"vllm-project/vllm","pin":PIN,
                      "chronology":"actualNov14UTC04:01:10/BJT12:01:10 multimodal graph cumulative-padding correction",
                      "license":"Apache-2.0 rights retained,no implementation mirror",
                      "verified":"default cumulative keys repeat terminal;explicit overrides first;FlashInfer Q/Ksections pad independently;scalars bypass;shared missing values error;oversize eager fallback",
                      "scope":"original1-D metadata/write-coverage model,not CUDAgraph/FlashInfer execution"},
            "case":matrix[0],"matrix":matrix,
            "summary":{"cases":len(matrix),"before_invalid_cases":sum(x["before"]["status"]!="valid_policy" for x in matrix),
                       "after_invalid_cases":sum(x["after"]["status"]!="valid_policy" for x in matrix),
                       "primary_before_word_misses":matrix[0]["before"]["word_misses"],
                       "primary_words":DEFAULT["capacity"],"primary_logical_bytes":4*DEFAULT["capacity"]},
            "packed_case":matrix[4],"latency":latency,"native_cuda_graph_verified":None,
            "native_flashinfer_attention_verified":None,"native_allocator_peak_bytes":None,"browser_visual_acceptance":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.measure:(HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text()))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:raise SystemExit(f"stale artifact:{file}")
        else:file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round160 sealed artifacts match" if a.check else "Wrote Round160original artifacts")


if __name__=="__main__":main()
