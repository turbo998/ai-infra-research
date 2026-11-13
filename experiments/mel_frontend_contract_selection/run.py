"""Original scalar mel-filter/projection contracts; no native ASR backend execution."""

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "880561aabb68be86c4c0f1c37ee5281e07ed1439"
MAX_EXACT = 2**53-1
DEFAULT = {"samples":16000,"frame_length":400,"hop_length":160,"fft_length":512,
           "mel_filters":80,"center":1,"batch":8,"element_bytes":4}


def f32(value):
    return struct.unpack("<f",struct.pack("<f",value))[0]


def mel_frequency(hz,scale):
    if scale=="htk":
        return 2595*math.log10(1+hz/700)
    if scale=="slaney":
        return hz/(200/3) if hz<1000 else 15+math.log(hz/1000)/(math.log(6.4)/27)
    raise ValueError("original model supports HTK or Slaney only")


def hertz(mel,scale):
    if scale=="htk":
        return 700*(10**(mel/2595)-1)
    if scale=="slaney":
        return mel*(200/3) if mel<15 else 1000*math.exp((mel-15)*(math.log(6.4)/27))
    raise ValueError("original model supports HTK or Slaney only")


def triangle(frequency,left,center,right):
    if not left<center<right:
        raise ValueError("strictly increasing triangle edges required")
    if frequency<=left or frequency>=right:
        return 0.0
    return (frequency-left)/(center-left) if frequency<=center else (right-frequency)/(right-center)


def filters(sample_rate=16000,fft_length=512,count=40,scale="slaney",normalize=True,cast_before=True):
    if (type(sample_rate) is not int or not 2<=sample_rate<=MAX_EXACT or type(fft_length) is not int or
            fft_length<2 or fft_length%2 or type(count) is not int or count<1 or
            count*(fft_length//2+1)>100000):
        raise ValueError("bounded positive filter count, rate and even FFT size required")
    if type(normalize) is not bool or type(cast_before) is not bool:
        raise ValueError("explicit Boolean normalization/rounding choices required")
    upper = mel_frequency(sample_rate/2,scale)
    edges = [hertz(upper*i/(count+1),scale) for i in range(count+2)]
    result = []
    for band in range(count):
        left,center,right = edges[band:band+3]
        factor = 2/(right-left) if normalize else 1
        row = []
        for bin_id in range(fft_length//2+1):
            weight = triangle(bin_id*sample_rate/fft_length,left,center,right)
            row.append(f32((f32(weight) if cast_before else weight)*factor))
        if not any(row):
            raise ValueError("selected original mel band has no sampled FFT support")
        result.append(row)
    return result


def compress(bank):
    return [[(index,value) for index,value in enumerate(row) if value] for row in bank]


def project(bank,frames):
    width = len(bank[0])
    if any(len(row)!=width for row in bank) or any(len(frame)!=width for frame in frames):
        raise ValueError("filter-major matrix and frame power widths must match")
    return [[sum(weight*value for weight,value in zip(row,frame)) for row in bank] for frame in frames]


def project_sparse(sparse,frames):
    return [[sum(weight*frame[index] for index,weight in row) for row in sparse] for frame in frames]


def power_db(power):
    if type(power) not in (int,float) or not math.isfinite(power) or power<=0:
        raise ValueError("positive finite power required; production floor policy is not modeled")
    return 10*math.log10(power)


def estimate_budget(p):
    for key in DEFAULT:
        minimum = 0 if key in ("samples","center") else 1
        if type(p[key]) is not int or not minimum<=p[key]<=MAX_EXACT:
            raise ValueError(f"{key} requires bounded exact integer >= {minimum}")
    if p["center"] not in (0,1) or p["element_bytes"] not in (2,4,8):
        raise ValueError("center must be 0/1; selected output width must be 2/4/8")
    if p["fft_length"]%2 or p["frame_length"]>p["fft_length"]:
        raise ValueError("selected FFT must be even and no smaller than frame")
    if p["samples"]<p["frame_length"] and not p["center"]:
        raise ValueError("original uncentered count requires at least one full frame")
    if p["center"] and p["samples"]<2:
        raise ValueError("original centered reflect selection requires at least two samples")
    padded = p["samples"]+2*(p["frame_length"]//2)*p["center"]
    frames = 1+(padded-p["frame_length"])//p["hop_length"]
    bins = p["fft_length"]//2+1
    result = {"frame_count":frames,"frequency_bins":bins,
              "selected_filter_float32_bytes":bins*p["mel_filters"]*4,
              "selected_power_spectrum_bytes":p["batch"]*frames*bins*p["element_bytes"],
              "selected_mel_output_bytes":p["batch"]*frames*p["mel_filters"]*p["element_bytes"]}
    if padded>MAX_EXACT or any(value>MAX_EXACT for value in result.values()):
        raise ValueError("result exceeds exact arithmetic range")
    return result


def measure():
    bank = filters()
    sparse = compress(bank)
    rows = []
    for count in (1,4,16,64):
        frames = [[((i+3*j)%17+1)/7 for i in range(257)] for j in range(count)]
        functions = (lambda:project_sparse(sparse,frames),lambda:project(bank,frames))
        assert functions[0]()==functions[1]()
        samples = [[],[]]
        for _ in range(2):
            for function in functions:
                function()
        for repeat in range(7):
            for index in ((0,1) if repeat%2==0 else (1,0)):
                started = time.perf_counter_ns()
                for _ in range(3):
                    functions[index]()
                samples[index].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"frames":count,"nonzero_samples_ms":samples[0],"dense_samples_ms":samples[1],
                     "nonzero_median_ms":statistics.median(samples[0]),
                     "dense_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original Python nonzero-weight projection versus dense scalar projection",
            "excludes":"filter construction/compression, waveform decode, STFT, resampling, native libraries and GPU",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows}


def build_report(latency):
    bank = filters()
    late = filters(cast_before=False)
    other = filters(scale="htk")
    bits = lambda value:struct.unpack("<I",struct.pack("<f",value))[0]
    differences = [abs(bits(a)-bits(b)) for row_a,row_b in zip(bank,late) for a,b in zip(row_a,row_b)]
    frame = [[(i%7+1)/3 for i in range(257)]]
    slaney,htk = project(bank,frame)[0],project(other,frame)[0]
    return {"schema_version":1,"round":116,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"huggingface/transformers","pin":PIN,"date_utc":"2026-11-12T10:35:44Z",
                      "license":"Apache-2.0","recent_change":"audio dependency slimming; model filter construction uses explicit float32 Slaney bank and transpose"},
            "case":{"sample_rate":16000,"fft_length":512,"mel_filters":40,"frequency_bins":257,
                    "matrix_entries":10280,"nonzero_entries":sum(bool(v) for row in bank for v in row),
                    "cast_order_different_entries":sum(bool(v) for v in differences),
                    "cast_order_max_ulp":max(differences),
                    "slaney_power_first_three":slaney[:3],"htk_power_first_three":htk[:3],
                    "max_scale_power_difference":max(abs(a-b) for a,b in zip(slaney,htk)),
                    "power_four_db":power_db(4),"wrong_twenty_log_power_four":20*math.log10(4),
                    "orientation":"original filter-major [mel][frequency]; source helper returns [frequency][mel]"},
            "variants":[{"frame_length":length,"center":center,
                         "outputs":estimate_budget({**DEFAULT,"frame_length":length,"center":center})}
                        for length,center in ((400,1),(401,1),(400,0))],
            "default_budget":{"inputs":DEFAULT,"outputs":estimate_budget(DEFAULT)},
            "latency":latency,"native_filter_bit_exactness":None,"native_backend_latency_ms":None,
            "asr_quality":None,"peak_hbm_bytes":None,"browser_visual_check":False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure",action="store_true")
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report = build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for path,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if args.check:
            if path.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact: {path}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text,encoding="utf-8")
    print("Round 116 artifacts match." if args.check else "Wrote Round 116 artifacts.")


if __name__=="__main__":
    main()
