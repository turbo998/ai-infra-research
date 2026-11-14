"""Original finite-digest HLL registers and classic teaching estimator."""

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="ee8e7f70152404649ac947157fd662d11b4a00d5"
DEFAULT={"precision_bits":14,"register_bits":6,"shards":64}


def register_rank(digest,precision=4,hash_bits=16):
    if type(precision) is not int or type(hash_bits) is not int or not 1<=precision<hash_bits<=32 or type(digest) is not int or not 0<=digest<2**hash_bits:
        raise ValueError("bounded exact digest/precision/hash width required")
    index=digest&(2**precision-1);tail=digest>>precision;rank=1
    while rank<=hash_bits-precision and not tail&1:
        rank+=1;tail>>=1
    return index,rank


def sketch(digests,precision=4,hash_bits=16):
    register_rank(0,precision,hash_bits)
    if precision>16:
        raise ValueError("original materialized sketch limit is precision16")
    registers=[0]*(2**precision)
    for digest in digests:
        index,rank=register_rank(digest,precision,hash_bits)
        registers[index]=max(registers[index],rank)
    return {"precision":precision,"hash_bits":hash_bits,"registers":registers}


def merge(states):
    if not states:
        raise ValueError("at least one compatible sketch required")
    p,h=states[0]["precision"],states[0]["hash_bits"]
    register_rank(0,p,h)
    if p>16:
        raise ValueError("original materialized sketch limit is precision16")
    for state in states:
        if state["precision"]!=p or state["hash_bits"]!=h or len(state["registers"])!=2**p or any(type(v) is not int or not 0<=v<=h-p+1 for v in state["registers"]):
            raise ValueError("incompatible precision/hash/rank register contract")
    return {"precision":p,"hash_bits":h,
            "registers":[max(state["registers"][i] for state in states) for i in range(2**p)]}


def classic_estimate(state):
    registers=merge([state])["registers"];m=len(registers)
    alpha={16:.673,32:.697,64:.709}.get(m,.7213/(1+1.079/m))
    harmonic=sum(2.0**(-v) for v in registers)
    raw=alpha*m*m/harmonic;zeros=registers.count(0)
    small=m*math.log(m/zeros) if zeros else None
    use_small=raw<=2.5*m and zeros>0
    return {"raw_classic_estimate":raw,"empty_registers":zeros,
            "linear_counting_estimate":small,"selected_teaching_estimate":small if use_small else raw,
            "selected_mode":"linear_counting" if use_small else "classic_raw",
            "not_native":"not Redis Ertl histogram estimator;finite-range correction excluded"}


def original_digest(identity):
    if type(identity) is not int or not 0<=identity<65536:
        raise ValueError("bounded exact original item identity required")
    return (identity*40503+17)%65536


def independent_register_oracle(digests,p=4,h=16):
    output=[]
    for index in range(2**p):
        ranks=[]
        for value in digests:
            if value%2**p==index:
                bits=f"{value//2**p:0{h-p}b}"[::-1]
                ranks.append(len(bits)-len(bits.lstrip("0"))+1)
        output.append(max(ranks,default=0))
    return output


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int for v in p.values()) or not 4<=p["precision_bits"]<=20 or not 1<=p["register_bits"]<=8 or not 1<=p["shards"]<=1000000:
        raise ValueError("bounded exact precision/register width/shards required")
    registers=2**p["precision_bits"];maxrank=64-p["precision_bits"]+1
    if 2**p["register_bits"]-1<maxrank:
        raise ValueError("selected register width cannot encode selected64bit hash max rank")
    payload=(registers*p["register_bits"]+7)//8
    return {"selected_register_count":registers,"selected_dense_register_payload_bytes":payload,
            "selected_all_shards_dense_payload_bytes":payload*p["shards"],
            "ideal_uniform_hash_asymptotic_rse":1.04/math.sqrt(registers),
            "note":"ideal-hash asymptotic standard error,not a per-case bound;metadata/sparse encoding/native allocations excluded"}


def timing_paths(n):
    digests=[original_digest(i) for i in range(n)]
    def bitwise():
        regs=[0]*16
        for d in digests:
            i=d&15;x=d>>4;rank=(x&-x).bit_length() if x else 13
            regs[i]=max(regs[i],rank)
        return regs
    def strings():
        regs=[0]*16
        for d in digests:
            i=d%16;s=f"{d//16:012b}"[::-1];rank=len(s)-len(s.lstrip("0"))+1
            regs[i]=max(regs[i],rank)
        return regs
    return bitwise,strings


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
        rows.append({"digests":n,"bitwise_samples_ms":samples[0],"string_samples_ms":samples[1],
                     "bitwise_median_ms":statistics.median(samples[0]),"string_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"original precomputed16bitdigest register update,bitwise rank versus binary-string rank",
            "excludes":"hashing,estimator,merge,Redis/native Murmur/Ertl,SIMD,sparse conversion,network","rows":rows}


def build_report(latency):
    a=list(range(20));b=list(range(10,30));da=list(map(original_digest,a));db=list(map(original_digest,b))
    sa,sb=sketch(da),sketch(db);union=merge([sa,sb]);direct=sketch(da+db)
    return {"schema_version":1,"round":136,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"redis/redis","pin":PIN,"license":"sourceheader RSALv2/SSPLv1/AGPLv3",
                      "verified":"low-bit index,remaining trailing-zero rank plus1 with sentinel,register max merge,native Ertl histogram estimator",
                      "chronology":"Nov13treechange improves SORT allocations,not new HLL",
                      "model_hash":"original16bit affine permutation,not Murmur/distribution guarantee",
                      "model_estimator":"classic raw+linearcounting teaching fixture,not native Redis Ertl"},
            "case":{"A_items":a,"B_items":b,"A_unique":len(set(a)),"B_unique":len(set(b)),"overlap_unique":len(set(a)&set(b)),
                    "true_union_unique":len(set(a)|set(b)),"wrong_sum_exact_counts":len(set(a))+len(set(b)),
                    "A_sketch":sa,"B_sketch":sb,"merged":union,"direct_union":direct,"oracle_union_registers":independent_register_oracle(da+db),
                    "merged_estimate":classic_estimate(union),"sum_local_teaching_estimates":classic_estimate(sa)["selected_teaching_estimate"]+classic_estimate(sb)["selected_teaching_estimate"],
                    "reversed_merge_equal":merge([sb,sa])==union,"duplicate_merge_equal":merge([sa,sa,sb])==union,
                    "zero_tail_rank":list(register_rank(0)),
                    "all_zero_sketch_estimate":classic_estimate(sketch([])),
                    "not_available":"exact membership,per-key deletion,compatible cross-hash merge or guaranteed error"},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_redis_latency_ms":None,"native_pfcount_result":None,"production_estimation_error":None,"native_memory_bytes":None}


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
    print("Round136 sealed artifacts match" if a.check else "Wrote Round136 original artifacts")


if __name__=="__main__":
    main()
