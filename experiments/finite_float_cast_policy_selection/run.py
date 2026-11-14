"""Original exact finite FP4/FP6 conversion policy;no native ONNX casting."""

import argparse
import bisect
from fractions import Fraction
import json
from pathlib import Path
import platform
import re
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="290eaba8185c0ed1ba6abda16f6f16d562a42da5"
FORMATS={"FLOAT4E2M1":(2,1,1),"FLOAT6E2M3":(2,3,1),"FLOAT6E3M2":(3,2,3)}
DEFAULT={"formats":list(FORMATS),"tokens":["0.75","5","+inf","nan"]}
DECIMAL=re.compile(r"^[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d{1,2})?$")


def integer(value,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError("bounded exact integer required")
    return value


def ticks_text(ticks,negative=False):
    whole,remainder=divmod(ticks,16)
    value=str(whole)+("."+f"{remainder*625:04d}".rstrip("0") if remainder else "")
    return ("-" if negative else "")+value


def decode(raw,fmt):
    if fmt not in FORMATS:
        raise ValueError("supported original finite format required")
    e,m,bias=FORMATS[fmt];sign_bit=1<<(e+m);integer(raw,0,2*sign_bit-1)
    negative=bool(raw&sign_bit);code=raw&(sign_bit-1);exponent=code>>m;mantissa=code&((1<<m)-1)
    scale_exp=(exponent-bias) if exponent else (1-bias)
    scale=Fraction(1<<scale_exp) if scale_exp>=0 else Fraction(1,1<<-scale_exp)
    value=Fraction((1<<m)+mantissa if exponent else mantissa,1<<m)*scale
    tick_value=value*16
    if tick_value.denominator!=1:
        raise RuntimeError("original finite format no longer fits selected exact tick lattice")
    ticks=tick_value.numerator
    return {"raw_code":raw,"value":ticks_text(ticks,negative),"magnitude_ticks":ticks,
            "negative":negative,"negative_zero":negative and ticks==0}


def book(fmt):
    e,m,bias=FORMATS[fmt];sign=1<<(e+m)
    ticks=[decode(code,fmt)["magnitude_ticks"] for code in range(sign)]
    return {"format":fmt,"sign_bit":sign,"total_bits":1+e+m,"exponent_bits":e,"mantissa_bits":m,"bias":bias,
            "raw_patterns":2*sign,"nan_encodings":0,"infinity_encodings":0,"positive_ticks":ticks,
            "minimum_subnormal":ticks_text(ticks[1]),"maximum":ticks_text(ticks[-1])}


def parse(token):
    if not isinstance(token,str) or not 1<=len(token)<=32:
        raise ValueError("bounded original decimal/special token string required")
    lowered=token.lower()
    if lowered=="nan":
        return "nan",None,False
    if lowered in ("inf","+inf","-inf"):
        return "inf",None,lowered.startswith("-")
    if not DECIMAL.fullmatch(token):
        raise ValueError("original decimal or explicit nan/inf token required")
    value=Fraction(token)
    if abs(value)>10**6:
        raise ValueError("original UI finite magnitude bound exceeded")
    return "finite",value,token.startswith("-")


def ratio(value):
    return {"numerator":str(value.numerator),"denominator":str(value.denominator)}


def nearest(target,ticks):
    pos=bisect.bisect_left(ticks,target)
    candidates={max(0,pos-1),min(pos,len(ticks)-1)}
    return min(candidates,key=lambda code:(abs(ticks[code]-target),code&1,code))


def convert(token,fmt):
    kind,value,negative=parse(token);table=book(fmt)
    if kind=="nan":
        return {"input":token,"format":fmt,"status":"blocked_nan_policy","raw_code":None,"value":None,
                "negative_zero":None,"absolute_error_ratio":None,
                "policy_error":"NaN cast is unspecified;original policy requires explicit handling"}
    ticks=table["positive_ticks"]
    code=len(ticks)-1 if kind=="inf" else nearest(abs(value)*16,ticks)
    raw=code|(table["sign_bit"] if negative else 0);result=decode(raw,fmt)
    decoded=Fraction(result["magnitude_ticks"],16)*(-1 if negative else 1)
    error=None if kind=="inf" else abs(value-decoded)
    if kind=="inf":
        status="documented_infinity_saturation"
    elif abs(value)*16>ticks[-1]:
        status="finite_saturation"
    elif result["magnitude_ticks"]==0 and value:
        status="underflow_to_signed_zero"
    else:
        status="exact" if error==0 else "rne_rounded"
    return {"input":token,"format":fmt,"status":status,"raw_code":raw,"value":result["value"],
            "negative_zero":result["negative_zero"],"absolute_error_ratio":None if error is None else ratio(error),"policy_error":None}


def evaluate(p):
    if not isinstance(p,dict) or set(p)!={"formats","tokens"}:
        raise ValueError("exact original formats/tokens input required")
    formats=p["formats"];tokens=p["tokens"]
    if not isinstance(formats,list) or not 1<=len(formats)<=3 or any(not isinstance(v,str) or v not in FORMATS for v in formats) or len(set(formats))!=len(formats):
        raise ValueError("one to three unique supported finite formats required")
    if not isinstance(tokens,list) or not 1<=len(tokens)<=32:
        raise ValueError("one to thirty-two original input tokens required")
    for token in tokens:
        parse(token)
    rows=[convert(token,fmt) for fmt in formats for token in tokens];summary=[]
    for fmt in formats:
        finite=[row for row in rows if row["format"]==fmt and row["absolute_error_ratio"] is not None]
        errors=[Fraction(int(row["absolute_error_ratio"]["numerator"]),int(row["absolute_error_ratio"]["denominator"])) for row in finite]
        summary.append({"format":fmt,"finite_error_observations":len(errors),
                        "mean_absolute_error_ratio":ratio(sum(errors)/len(errors)) if errors else None})
    return {"rows":rows,"summary":summary,"blocked_nan_rows":sum(row["status"]=="blocked_nan_policy" for row in rows),
            "scope":"original exact rational inputs and saturating RNE policy;no float32 pre-rounding,scales,calibration,ONNX/reference/backend/GPU casts"}


def payload(count):
    integer(count,0,10**6)
    return {"elements":count,"selected_raw_dtype_payload_bytes":{fmt:(count*book(fmt)["total_bits"]+7)//8 for fmt in FORMATS},
            "scope":"nominal dense raw dtype payload only;not an executed tensor,metadata,scales,runtime peak or wire bytes"}


def timing_paths(n):
    ticks=book("FLOAT6E3M2")["positive_ticks"];inputs=[i%513 for i in range(n)]
    def indexed():
        return [nearest(v,ticks) for v in inputs]
    def scanned():
        return [min(range(len(ticks)),key=lambda code:(abs(ticks[code]-v),code&1,code)) for v in inputs]
    return indexed,scanned


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
        rows.append({"positive_tick_queries":n,"bisect_samples_ms":samples[0],"scan_samples_ms":samples[1],
                     "bisect_median_ms":statistics.median(samples[0]),"scan_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original positive integer-tick nearest-code lookup,bisected candidates versus32candidate scan,both RNE tie selection",
            "excludes":"table/input construction,decimal parsing,signed-zero/NaN/Inf handling,native ONNX/GPU casting and numerical calibration;timer exceeds UI bound","rows":rows}


def build_report(latency):
    robustness={"formats":list(FORMATS),"tokens":["0","-0","0.25","-0.25","1.25","2.5","10","25"]}
    return {"schema_version":1,"round":148,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"onnx/onnx","pin":PIN,"license":"Apache-2.0 source rights preserved;no implementation mirrored",
                      "verified":"finite E2M1/E2M3/E3M2 bitfields,bias,subnormals,RNE,signed zero,saturation;NaN conversions explicitly unspecified",
                      "chronology":"Nov12UTC16:05 = Nov13BJT00:05documentation/test clarification;not introduction of FP4/FP6 or a new native algorithm",
                      "scope":"original exact rational saturating conversion with reject-NaN policy;not native float32 pre-rounding or backend error guarantee"},
            "formats":[book(fmt) for fmt in FORMATS],"case":{"inputs":DEFAULT,"outputs":evaluate(DEFAULT)},
            "robustness":{"inputs":robustness,"outputs":evaluate(robustness)},"selected_payload":payload(9),"latency":latency,
            "native_nan_cast_result":None,"native_onnx_cast_verified":None,"gpu_cast_latency_ms":None,
            "native_runtime_peak_memory_bytes":None,"model_quality_metrics":None}


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
    print("Round148 sealed artifacts match" if args.check else "Wrote Round148 original artifacts")


if __name__=="__main__":
    main()
