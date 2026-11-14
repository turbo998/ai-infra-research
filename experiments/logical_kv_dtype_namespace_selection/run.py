"""Original typed object-cache model;no SGLang,Mooncake or native KV casts."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform
import re
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="a05c729c4eb233c73b22f7a9289bcd11868d8cee"
FORMATS={"float8_e4m3fn":(4,3,7,1,True),"float8_e5m2":(5,2,15,1,False),
         "float16":(5,10,15,2,False),"bfloat16":(8,7,127,2,False)}
DEFAULT={"model_id":"synthetic-model","storage_tag":None,"extra_backend_tag":"layout-v1","page_id":"page-7",
         "writes":[{"logical_dtype":"float8_e4m3fn","physical_dtype":"uint8","payload_hex":"3c00"},
                   {"logical_dtype":"float8_e5m2","physical_dtype":"uint8","payload_hex":"4000"},
                   {"logical_dtype":"float16","physical_dtype":"float16","payload_hex":"003e"},
                   {"logical_dtype":"bfloat16","physical_dtype":"bfloat16","payload_hex":"803f"}]}


def token(v,nullable=False):
    if nullable and v is None:
        return
    if not isinstance(v,str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,64}",v):
        raise ValueError("bounded original ASCII metadata token required")


def ratio(v):
    return {"numerator":str(v.numerator),"denominator":str(v.denominator)}


def probe(payload,dtype):
    e,m,bias,width,finite_only=FORMATS[dtype]
    if len(payload)<width:
        return {"status":"no_complete_scalar","ratio":None,"negative_zero":None}
    raw=int.from_bytes(payload[:width],"little");negative=bool(raw>>(e+m))
    exponent=(raw>>m)&((1<<e)-1);mantissa=raw&((1<<m)-1)
    if exponent==(1<<e)-1 and (not finite_only or mantissa==(1<<m)-1):
        return {"status":"nan" if finite_only or mantissa else "infinity","ratio":None,"negative_zero":None}
    power=(exponent if exponent else 1)-bias-m
    magnitude=Fraction(((1<<m)+mantissa) if exponent else mantissa)*(Fraction(1<<power) if power>=0 else Fraction(1,1<<-power))
    return {"status":"finite","ratio":ratio(-magnitude if negative else magnitude),"negative_zero":negative and magnitude==0}


def key(p,dtype):
    return json.dumps(["original-v1",p["model_id"],p["storage_tag"],dtype,p["extra_backend_tag"],p["page_id"]],separators=(",",":"))


def read(cache,cache_key,request):
    if cache_key not in cache:
        return {"status":"miss","payload_hex":None,"stored_dtype_writer_witness":None,
                "dtype_writer_witness_matches":None,"payload_writer_witness_matches":None,
                "reader_probe":None,"finite_probe_absolute_error":None}
    stored=cache[cache_key];same=stored["logical_dtype"]==request["logical_dtype"]
    payload=bytes.fromhex(stored["payload_hex"]);expected=bytes.fromhex(request["payload_hex"])
    actual=probe(payload,request["logical_dtype"]);wanted=probe(expected,request["logical_dtype"]);error=None
    if actual["ratio"] is not None and wanted["ratio"] is not None:
        a=actual["ratio"];b=wanted["ratio"]
        error=ratio(abs(Fraction(int(a["numerator"]),int(a["denominator"]))-Fraction(int(b["numerator"]),int(b["denominator"]))))
    return {"status":"length_mismatch_by_writer_witness" if len(payload)!=len(expected) else "hit" if same else "foreign_hit_by_writer_witness",
            "payload_hex":stored["payload_hex"],"stored_dtype_writer_witness":stored["logical_dtype"],
            "dtype_writer_witness_matches":same,"payload_writer_witness_matches":payload==expected,
            "reader_probe":actual,"finite_probe_absolute_error":error}


def evaluate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original namespace/write fields required")
    token(p["model_id"],True);token(p["storage_tag"],True);token(p["extra_backend_tag"],True);token(p["page_id"])
    if not isinstance(p["writes"],list) or not 1<=len(p["writes"])<=4:
        raise ValueError("original1..4unique logical dtype writes required")
    seen=set();writes=[]
    for w in p["writes"]:
        if not isinstance(w,dict) or set(w)!={"logical_dtype","physical_dtype","payload_hex"}:
            raise ValueError("exact original logical/physical dtype and hex payload required")
        dtype=w["logical_dtype"]
        if not isinstance(dtype,str) or dtype not in FORMATS or dtype in seen:
            raise ValueError("resolved unique original logical dtype required;no auto/backing fallback")
        if w["physical_dtype"] not in ("uint8","float16","bfloat16"):
            raise ValueError("original physical dtype label required")
        value=w["payload_hex"]
        if not isinstance(value,str) or not re.fullmatch(r"(?:[0-9a-fA-F]{2}){1,32}",value):
            raise ValueError("bounded complete original hex-byte pairs required")
        data=bytes.fromhex(value)
        if len(data)%FORMATS[dtype][3]:
            raise ValueError("original scalar payload must contain complete words")
        seen.add(dtype);writes.append({**w,"payload_hex":data.hex()})
    legacy={};physical={};typed={}
    for w in writes:
        legacy[key(p,None)]=w;physical[key(p,"backing:"+w["physical_dtype"])]=w;typed[key(p,"logical:"+w["logical_dtype"])]=w
    rows=[{"logical_dtype":w["logical_dtype"],"physical_dtype":w["physical_dtype"],
           "typed_key":key(p,"logical:"+w["logical_dtype"]),"expected_probe_writer_witness":probe(bytes.fromhex(w["payload_hex"]),w["logical_dtype"]),
           "legacy":read(legacy,key(p,None),w),"physical_only":read(physical,key(p,"backing:"+w["physical_dtype"]),w),
           "typed":read(typed,key(p,"logical:"+w["logical_dtype"]),w)} for w in writes]
    first=writes[0];cold=read({},key(p,"logical:"+first["logical_dtype"]),first)
    fresh=read({key(p,"logical:"+first["logical_dtype"]):first},key(p,"logical:"+first["logical_dtype"]),first)
    return {"rows":rows,"legacy_objects":len(legacy),"physical_only_objects":len(physical),"typed_objects":len(typed),
            "legacy_foreign_reads":sum(not r["legacy"]["dtype_writer_witness_matches"] for r in rows),
            "physical_only_foreign_reads":sum(not r["physical_only"]["dtype_writer_witness_matches"] for r in rows),
            "typed_foreign_reads":sum(not r["typed"]["dtype_writer_witness_matches"] for r in rows),
            "selected_stored_payload_bytes":{"legacy":sum(len(bytes.fromhex(w["payload_hex"])) for w in legacy.values()),
                                            "physical_only":sum(len(bytes.fromhex(w["payload_hex"])) for w in physical.values()),
                                            "typed":sum(len(bytes.fromhex(w["payload_hex"])) for w in typed.values())},
            "migration":{"legacy_exists":True,"typed_before_repopulate":cold,"fallback_used":False,"typed_after_repopulate":fresh},
            "native_sglang_key_compatibility":None,"native_kv_accuracy":None}


def timing_paths(n):
    p=DEFAULT;dtype="logical:float8_e4m3fn"
    prefix=json.dumps(["original-v1",p["model_id"],p["storage_tag"],dtype,p["extra_backend_tag"]],separators=(",",":"))[:-1]+","
    pages=[f"page-{i}" for i in range(n)]
    def hoisted():
        return [prefix+json.dumps(page)+"]" for page in pages]
    def rebuilt():
        return [key({**p,"page_id":page},dtype) for page in pages]
    return hoisted,rebuilt


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=timing_paths(n)
        if funcs[0]()!=funcs[1]():
            raise RuntimeError("original namespace serializers disagree")
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
        rows.append({"pages":n,"hoisted_samples_ms":samples[0],"rebuilt_samples_ms":samples[1],
                     "hoisted_median_ms":statistics.median(samples[0]),"rebuilt_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original canonical JSON keys,reused namespace prefix versus full descriptor serialization per page",
            "excludes":"page/input setup,payload decoding,hashes,msgspec/SGLang/Mooncake/storage/GPU;timer exceeds UIbounds","rows":rows}


def build_report(latency):
    p={**DEFAULT,"model_id":None,"storage_tag":"envelope-v2","extra_backend_tag":"layout-v2"}
    return {"schema_version":1,"round":153,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"sgl-project/sglang","pin":PIN,"license":"Apache-2.0 rights preserved;no implementation mirror",
                      "chronology":"actualNov14UTC05:22/BJT13:22 HiCache logical dtype key correction",
                      "verified":"resolved token-view logical dtype,not uint8 backing;storage_format_tag retained;DCP/non-DCP cold migration with no untyped fallback",
                      "scope":"original canonical JSON object cache and exact scalar probes;not SGLang key encoding,serialization,FP8 casts or KV accuracy"},
            "case":{"inputs":DEFAULT,"outputs":evaluate(DEFAULT)},"robustness":{"inputs":p,"outputs":evaluate(p)},
            "same_byte_probe":{"hex":"3c","float8_e4m3fn":probe(b"\x3c","float8_e4m3fn"),"float8_e5m2":probe(b"\x3c","float8_e5m2")},
            "latency":latency,"native_sglang_verified":None,"native_mooncake_verified":None,
            "native_cache_hit_rate":None,"native_model_quality":None,"native_storage_latency_ms":None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
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
    print("Round153 sealed artifacts match" if args.check else "Wrote Round153 original artifacts")


if __name__=="__main__":
    main()
