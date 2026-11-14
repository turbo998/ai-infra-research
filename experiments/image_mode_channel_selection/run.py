"""Original post-decode mode normalization;no image codec/reference execution."""

import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="3c41120b9c69657a1dd13407b080bd451ed06c9c"
CHANNELS={"1":1,"L":1,"LA":2,"P":1,"RGB":3,"RGBA":4}
DEFAULT={"mode":"RGBA","height":1,"width":2,"pixels":[9,70,220,0,190,25,14,255],
         "palette":None,"pixel_format":"BGR"}


def integer(value,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError("bounded exact integer required")
    return value


def rgb(p):
    if not isinstance(p,dict) or set(p)!={"mode","height","width","pixels","palette","pixel_format"}:
        raise ValueError("exact original source-mode fixture required")
    if not isinstance(p["mode"],str) or p["mode"] not in CHANNELS or p["pixel_format"] not in ("RGB","BGR"):
        raise ValueError("supported original source mode and RGB/BGR target required")
    n=integer(p["height"],1,16)*integer(p["width"],1,16);mode=p["mode"];channels=CHANNELS[mode]
    if not isinstance(p["pixels"],list) or len(p["pixels"])!=n*channels:
        raise ValueError("original flat pixel count must match source-mode dimensions")
    for v in p["pixels"]:
        if mode=="1":
            if type(v) is not bool:
                raise ValueError("original bilevel fixture requires boolean values")
        else:
            integer(v,0,255)
    palette=p["palette"]
    if mode=="P":
        if not isinstance(palette,list) or not 1<=len(palette)<=256:
            raise ValueError("original palette RGB entries required")
        for colour in palette:
            if not isinstance(colour,list) or len(colour)!=3:
                raise ValueError("original palette entries must have three channels")
            for v in colour:
                integer(v,0,255)
        if any(v>=len(palette) for v in p["pixels"]):
            raise ValueError("original palette index out of bounds")
    elif palette is not None:
        raise ValueError("palette only applies to original P fixtures")
    result=[]
    for i in range(n):
        values=p["pixels"][i*channels:(i+1)*channels]
        if mode=="1":
            result.extend([255 if values[0] else 0]*3)
        elif mode in ("L","LA"):
            result.extend([values[0]]*3)
        elif mode=="P":
            result.extend(palette[values[0]])
        else:
            result.extend(values[:3])
    return result


def reverse_channels(values,channels=3):
    return [v for i in range(0,len(values),channels) for v in values[i:i+channels][::-1]]


def toy_score(values):
    return sum(2*values[i]+values[i+1]-values[i+2] for i in range(0,len(values),3))


def evaluate(p):
    normalized=rgb(p);h=p["height"];w=p["width"];mode=p["mode"];channels=CHANNELS[mode]
    output=normalized if p["pixel_format"]=="RGB" else reverse_channels(normalized)
    naive_shape=[h,w] if channels==1 else [h,w,channels]
    naive_dtype="bool" if mode=="1" else "uint8"
    naive_error=None;naive=list(p["pixels"])
    if p["pixel_format"]=="BGR":
        if channels==1:
            naive=None;naive_error="original naive raw fixture has no channel axis to reverse"
        else:
            naive=reverse_channels(naive,channels)
    return {"shape":[h,w,3],"dtype":"uint8","pixels":output,"logical_output_bytes":len(output),
            "source_mode_value_count":len(p["pixels"]),"canonical_rgb":normalized,
            "toy_rgb_score":toy_score(normalized),"toy_score_if_output_mislabeled_rgb":toy_score(output),
            "naive_shape":naive_shape,"naive_dtype":naive_dtype,"naive_pixels":naive,"naive_error":naive_error,
            "naive_matches_contract":naive_shape==[h,w,3] and naive_dtype=="uint8" and naive==output}


def fixtures():
    base={**DEFAULT,"pixels":[]}
    values={"1":[False,True],"L":[18,207],"LA":[18,0,207,255],"P":[0,1],
            "RGB":[9,70,220,190,25,14],"RGBA":DEFAULT["pixels"]}
    rows=[]
    for mode,pixels in values.items():
        for target in ("RGB","BGR"):
            p={**base,"mode":mode,"pixels":list(pixels),"pixel_format":target,
               "palette":[[9,70,220],[190,25,14]] if mode=="P" else None}
            rows.append({"inputs":p,"outputs":evaluate(p)})
    return rows


def timing_paths(n):
    data=bytes(v for i in range(n) for v in (i%251,(i*3)%251,(i*7)%251))
    def strided():
        out=bytearray(len(data));out[0::3]=data[2::3];out[1::3]=data[1::3];out[2::3]=data[0::3]
        return bytes(out)
    def scalar():
        out=bytearray(len(data))
        for i in range(0,len(data),3):
            out[i]=data[i+2];out[i+1]=data[i+1];out[i+2]=data[i]
        return bytes(out)
    return strided,scalar


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=timing_paths(n)
        if funcs[0]()!=funcs[1]():
            raise RuntimeError("original channel-order paths disagree")
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
        rows.append({"pixels":n,"logical_bytes":3*n,"strided_samples_ms":samples[0],"scalar_samples_ms":samples[1],
                     "strided_median_ms":statistics.median(samples[0]),"scalar_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original pre-normalized RGB bytes to copied BGR bytes,strided assignments versus scalar pixel loop",
            "excludes":"image opening,compressed codecs,source-mode/palette/alpha conversion,input preparation,NumPy/Pillow/ONNX,GPU upload/strides;timer exceeds UI bounds",
            "rows":rows}


def build_report(latency):
    alpha=copy.deepcopy(DEFAULT);alpha["pixels"][3]=255;alpha["pixels"][7]=0
    rows=fixtures()
    return {"schema_version":1,"round":150,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"onnx/onnx","pin":PIN,"license":"Apache-2.0 rights preserved;no implementation mirror",
                      "chronology":"Nov12UTC23:39 = Nov13BJT07:39 reference correction,reviewed Nov14;opset20 already existed",
                      "verified":"RGB/BGR converts decoded mode to RGB first;RGB default,HWC uint8;Grayscale path and image format acceptance unchanged",
                      "scope":"original six post-decode logical modes,not codecs,Pillow/reference conformance,colour management or native contiguous-layout guarantee"},
            "case":{"inputs":DEFAULT,"outputs":evaluate(DEFAULT)},"matrix":rows,
            "summary":{"cases":len(rows),"naive_misses":sum(not v["outputs"]["naive_matches_contract"] for v in rows)},
            "robustness":{"alpha_inputs":alpha,"alpha_outputs":evaluate(alpha),
                          "alpha_changes_output":evaluate(alpha)["pixels"]!=evaluate(DEFAULT)["pixels"],
                          "swapping_twice":reverse_channels(reverse_channels(evaluate(DEFAULT)["canonical_rgb"]))},
            "latency":latency,"native_decode_verified":None,"native_encoded_input_immutability_verified":None,
            "native_bgr_contiguous":None,"native_decode_latency_ms":None,"native_peak_memory_bytes":None,"model_quality_metrics":None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if args.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round150 sealed artifacts match" if args.check else "Wrote Round150 original artifacts")


if __name__=="__main__":
    main()
