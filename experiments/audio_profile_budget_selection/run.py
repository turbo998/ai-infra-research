"""Original integer audio-profile sizing; no waveform,feature extractor or model."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PIN="24d4c4949db5d6d8c3149eb3c8f2ddb1f2c0224d"
DEFAULT={"mel_sr":16000,"hop":160,"mel_dim":128,"audio_count":1,"override_samples":None,
         "time_markers":False,"seq_len":4096,"text_tokens":0}


def integer(x,lo=0,hi=10000000):
    return type(x) is int and lo<=x<=hi


def tokens(frames,fused=True):
    if fused:return (frames+7)//8
    for _ in range(3):frames=(frames+1)//2
    return frames


def plan(p,strategy="rate_aligned"):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):raise ValueError("Exactly the original audio sizing fields required")
    if p["mel_sr"] not in (8000,16000,22050,24000,44100,48000) or type(p["mel_sr"]) is not int or not integer(p["hop"],1,1024) or not integer(p["mel_dim"],1,256):
        raise ValueError("Known original sample rate and positive bounded hop/mel dimension required")
    if not integer(p["audio_count"],0,64) or not integer(p["seq_len"],1,131072) or not integer(p["text_tokens"],0,131072) or type(p["time_markers"]) is not bool:
        raise ValueError("Bounded item/context counts and explicit marker flag required")
    override=p["override_samples"]
    if override is not None and not integer(override):raise ValueError("Override is null or bounded nonnegative samples")
    if strategy not in ("rate_aligned","legacy_fixed"):raise ValueError("Known dummy sizing strategy required")
    maximum=p["mel_sr"]*30
    budget_frames=(maximum+p["hop"]-1)//p["hop"]
    budget=tokens(budget_frames)
    samples=maximum if strategy=="rate_aligned" else 16000
    warnings=[]
    if override:
        if override>samples:warnings.append("override_above_selected_max_ignored")
        samples=min(samples,override)
    frames=(samples+p["hop"]-1)//p["hop"]
    count=tokens(frames)
    digits=0
    if p["time_markers"]:
        full_seconds=(count*p["hop"]*8)//p["mel_sr"]
        digits=sum(len(str(second)) for second in range(2,full_seconds+1,2))
    n=p["audio_count"]
    span=n*(count+digits+2)
    return {"max_audio_seconds":30,"selected_samples":samples if n else 0,
            "declared_raw_mel_frames_per_item":frames if n else 0,
            "declared_audio_budget":({"audio":budget} if n else {}),
            "declared_audio_tokens_per_item":count if n else 0,"total_audio_tokens":n*count,
            "marker_digit_tokens_per_item":digits if n else 0,"audio_span_tokens":span,
            "context_ledger_fits":span+p["text_tokens"]<=p["seq_len"],
            "budget_shortfall_per_item":max(0,budget-count) if n else 0,
            "declared_unique_numpy_waveform_bytes":samples*8 if n else 0,
            "declared_float32_mel_batch_bytes":n*p["mel_dim"]*frames*4,
            "warnings":warnings,"native_extracted_frames":None,"native_peak_bytes":None,
            "native_audio_tokenizer_or_encoder_verified":None}


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        lengths=[1+(i*97)%10001 for i in range(n)]
        if [tokens(x) for x in lengths]!=[tokens(x,False) for x in lengths]:raise AssertionError("Ceiling equivalence")
        samples={True:[],False:[]}
        for mode in (True,False):
            for _ in range(2):[tokens(x,mode) for x in lengths]
        for repeat in range(7):
            for mode in ((True,False) if repeat%2==0 else (False,True)):
                start=time.perf_counter_ns()
                for _ in range(3):[tokens(x,mode) for x in lengths]
                samples[mode].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"items":n,"fused_samples_ms":samples[True],"three_stage_samples_ms":samples[False],
                     "fused_median_ms":statistics.median(samples[True]),"three_stage_median_ms":statistics.median(samples[False])})
    return {"provenance":"measured","scope":"Original Python scalar sizing only,no waveform/STFT/Torch/GPU",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows,
            "warmups":2,"repeats":7,"batches_per_sample":3,"native_profile_latency_ms":None}


def report(latency):
    variants=[("default thirty-second profile",{}),("time markers",{"time_markers":True}),
              ("short override",{"override_samples":16000}),("above-max override",{"override_samples":960000}),
              ("small context",{"seq_len":128}),("no audio",{"audio_count":0}),
              ("three shared-waveform items",{"audio_count":3})]
    return {"schema_version":1,"round":169,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"vllm-project/vllm","pin":PIN,"chronology":"actualNov14UTC07:15:27/BJT15:15:27 MOSS-Audio dummy budget",
                      "license":"Apache-2.0 rights retained,no implementation mirror",
                      "verified":"dummy mel_sr times30,ceil-budget frames,three stride2 lengths,override clamp/shared waveform and marker/span distinction",
                      "scope":"original integer sizing,not native waveform/STFT/tokenizer/Torch/GPU or peak profiling"},
            "case":{"inputs":DEFAULT,"current":plan(DEFAULT),"legacy":plan(DEFAULT,"legacy_fixed")},
            "matrix":[{"name":name,"inputs":{**DEFAULT,**changes},"outputs":plan({**DEFAULT,**changes})} for name,changes in variants],
            "latency":latency,"native_processing_test_executed":False,"visual_acceptance":None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--measure",action="store_true");ap.add_argument("--check",action="store_true");a=ap.parse_args()
    path=HERE/"latency.json"
    if a.measure or not path.exists():path.write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=report(json.loads(path.read_text(encoding="utf-8")))
    from experiments.audio_profile_budget_selection.render import outputs
    files={HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}
    for path,value in files.items():
        if a.check:
            if path.read_text(encoding="utf-8")!=value:raise AssertionError(f"Unsealed artifact:{path}")
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding="utf-8")
    print("Round169 sealed artifacts match" if a.check else "Rendered original audio-profile sizing")


if __name__=="__main__":main()
