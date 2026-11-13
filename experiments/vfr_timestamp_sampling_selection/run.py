"""Original integer-tick presentation-time lookup and clip-padding references."""

import argparse
import bisect
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "be5a80b79e19d928125a660f3c8f067af1fe6866"
MAX_SAFE = 2**53 - 1
DEFAULT = dict(clips=32, frames_per_clip=16, unique_targets=400,
               height=224, width=224, channels=3, scalar_bytes=1)


def integer(value, low=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= low


@dataclass(frozen=True)
class Timeline:
    pts: tuple
    end: int
    ticks_per_second: int = 1000

    def __post_init__(self):
        if (not isinstance(self.pts,tuple) or not self.pts or
                not all(isinstance(v,int) and not isinstance(v,bool) for v in self.pts) or
                any(a >= b for a,b in zip(self.pts,self.pts[1:])) or
                not isinstance(self.end,int) or isinstance(self.end,bool) or self.end <= self.pts[-1] or
                not integer(self.ticks_per_second,1)):
            raise ValueError("strictly increasing integer PTS, later end and positive tick rate required")


def validate_requests(timeline, requests):
    if any(not isinstance(value,int) or isinstance(value,bool) or
           not timeline.pts[0] <= value < timeline.end for value in requests):
        raise ValueError("requests must be integer ticks in the half-open stream interval")


def binary_lookup(timeline, requests):
    validate_requests(timeline,requests)
    return [bisect.bisect_right(timeline.pts,value)-1 for value in requests]


def interval_oracle(timeline, requests):
    validate_requests(timeline,requests)
    intervals = tuple(zip(timeline.pts,(*timeline.pts[1:],timeline.end)))
    return [next(i for i,(start,end) in enumerate(intervals) if start <= value < end)
            for value in requests]


def average_fps_lookup(timeline, requests, numerator, denominator):
    validate_requests(timeline,requests)
    if not integer(numerator,1) or not integer(denominator,1):
        raise ValueError("positive rational FPS required")
    indices = [(value-timeline.pts[0])*numerator//(timeline.ticks_per_second*denominator)
               for value in requests]
    if any(index >= len(timeline.pts) for index in indices):
        raise ValueError("header FPS maps a request beyond available frame indices")
    return indices


def restore_slots(indices):
    if any(not integer(index) for index in indices):
        raise ValueError("nonnegative integer frame identities required")
    unique = sorted(set(indices))
    rows = {index:i for i,index in enumerate(unique)}
    inverse = [rows[index] for index in indices]
    return dict(unique_targets=unique,output_to_unique=inverse,
                restored_slots=[unique[row] for row in inverse])


def clip_timestamps(timeline,start,count,step,policy):
    validate_requests(timeline,[start])
    if not integer(count,1) or not integer(step,1) or policy not in ("repeat_last","wrap","error"):
        raise ValueError("positive integer count/step and repeat_last/wrap/error required")
    valid = [start+i*step for i in range(count) if start+i*step < timeline.end]
    if len(valid) < count:
        if policy == "error":
            raise ValueError("clip extends beyond the half-open stream endpoint")
        if policy == "repeat_last":
            return valid+[valid[-1]]*(count-len(valid))
        return (valid*((count+len(valid)-1)//len(valid)))[:count]
    return valid


def estimate_budget(p):
    for key in ("clips","frames_per_clip","height","width","channels"):
        if not integer(p[key],1):
            raise ValueError(f"{key} must be a positive integer")
    if not integer(p["unique_targets"]) or p["unique_targets"] < 1:
        raise ValueError("unique_targets must be positive for a nonempty output")
    if p["scalar_bytes"] not in (1,2,4) or isinstance(p["scalar_bytes"],bool):
        raise ValueError("scalar_bytes must be 1, 2 or 4")
    slots = p["clips"]*p["frames_per_clip"]
    if p["unique_targets"] > slots:
        raise ValueError("unique_targets cannot exceed requested output slots")
    frame_bytes = p["height"]*p["width"]*p["channels"]*p["scalar_bytes"]
    result = dict(output_slots=slots,unique_target_count=p["unique_targets"],
                  repeated_output_slots=slots-p["unique_targets"],one_frame_bytes=frame_bytes,
                  output_tensor_bytes=slots*frame_bytes,
                  unique_target_tensor_payload_bytes=p["unique_targets"]*frame_bytes,
                  pts_and_duration_float64_bytes=slots*16,
                  output_index_int64_bytes=slots*8,
                  actual_decoded_frame_count=None,decoder_latency_ms=None,
                  peak_hbm_bytes=None,model_quality=None)
    if any(value > MAX_SAFE for value in result.values() if isinstance(value,int)):
        raise ValueError("budget exceeds exact JavaScript integer range")
    return result


def fixture_timeline(count):
    pts = [1000]
    for i in range(count-1):
        pts.append(pts[-1]+(20,40,120,60)[i%4])
    return Timeline(tuple(pts),pts[-1]+80)


def measure_latency():
    rows = []
    for count in (64,256,1024,4096):
        timeline = fixture_timeline(count)
        requests = [timeline.pts[0]+i*(timeline.end-timeline.pts[0]-1)//127 for i in range(128)]
        methods = (lambda: binary_lookup(timeline,requests),lambda: interval_oracle(timeline,requests))
        samples = [[],[]]
        for method in methods:
            for _ in range(2):
                method()
        for repeat in range(7):
            for index in ((0,1) if repeat%2 == 0 else (1,0)):
                start = time.perf_counter_ns()
                methods[index]()
                samples[index].append((time.perf_counter_ns()-start)/1e6)
        rows.append(dict(frames=count,requests=len(requests),
                         primary_samples_ms=samples[0],baseline_samples_ms=samples[1],
                         primary_median_ms=statistics.median(samples[0]),
                         baseline_median_ms=statistics.median(samples[1]),
                         index_mismatches=sum(a!=b for a,b in zip(methods[0](),methods[1]()))))
    data = dict(provenance="measured",scope="Original Python CPU integer-tick metadata lookup only",
                primary="Binary search in prevalidated immutable PTS tuple",
                baseline="Construct interval tuple and linearly scan each request",
                excludes="Index/scan construction, file I/O, codec decode, seek pre-roll and tensor conversion",
                hardware=platform.machine(),processor=platform.processor(),python=platform.python_version(),
                warmups=2,repeats=7,alternating_order=True,rows=rows,
                torchcodec_measured=False,decoder_latency_ms=None)
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def simulate():
    timeline = Timeline((1000,1040,1080,1200,1240),1400)
    requests = [1000,1039,1040,1080,1199,1200,1239,1240,1399]
    exact = binary_lookup(timeline,requests)
    approximate = average_fps_lookup(timeline,requests,25,2)
    duplicates = [1190,1100,1041,1080]
    slots = restore_slots(binary_lookup(timeline,duplicates))
    policies = {}
    for policy in ("repeat_last","wrap"):
        ticks = clip_timestamps(timeline,1160,5,80,policy)
        policies[policy] = dict(ticks=ticks,frame_indices=binary_lookup(timeline,ticks))
    raw = [1160+i*80 for i in range(5)]
    wrong_modulo = [timeline.pts[0]+(t-timeline.pts[0])%(timeline.end-timeline.pts[0]) for t in raw]
    short_tail = clip_timestamps(timeline,1190,4,300,"repeat_last")
    reordered_records = [(1000,"A"),(1080,"C"),(1040,"B"),(1240,"E"),(1200,"D")]
    presentation = sorted(reordered_records)
    variants = []
    for count in (3,7,17,31):
        tl = fixture_timeline(count)
        qs = list(tl.pts)+[tl.end-1,tl.pts[0]]
        a,b = binary_lookup(tl,qs),interval_oracle(tl,qs)
        variants.append(dict(frames=count,requests=len(qs),
                             index_mismatches=sum(a!=b for a,b in zip(a,b))))
    cfr = Timeline((1000,1040,1080,1120,1160),1200)
    cfr_queries = list(range(1000,1200))
    cfr_misses = sum(a!=b for a,b in zip(binary_lookup(cfr,cfr_queries),
                                       average_fps_lookup(cfr,cfr_queries,25,1)))
    if not (HERE/"latency.json").exists():
        raise FileNotFoundError("Run --measure once to seal original CPU timings")
    return dict(schema_version=1,round=107,provenance="synthetic",independently_reproduced=False,
                torchcodec_measured=False,
                source=dict(pin=PIN,date="2026-11-09",license="BSD-3-Clause",
                            capability="Low-level APIs graduated from BETA to stable, not a new VFR algorithm",
                            url=f"https://github.com/pytorch/torchcodec/blob/{PIN}/src/torchcodec/_core/SingleStreamDecoder.cpp"),
                timeline=dict(pts=timeline.pts,end=timeline.end,ticks_per_second=1000,
                              interval_ticks=[b-a for a,b in zip(timeline.pts,(*timeline.pts[1:],timeline.end))]),
                sampling=dict(request_ticks=requests,exact_indices=exact,approximate_indices=approximate,
                              average_fps_numerator=25,average_fps_denominator=2,
                              approximate_mismatches=sum(a!=b for a,b in zip(exact,approximate)),
                              denominator=len(requests),cfr_requests=len(cfr_queries),cfr_mismatches=cfr_misses,
                              wrong_ignore_origin_raw_indices=[t*25//2000 for t in requests]),
                duplicates=dict(request_ticks=duplicates,**slots,output_slot_count=len(duplicates),
                                unique_target_count=len(slots["unique_targets"]),
                                actual_decoded_frame_count=None),
                padding=dict(**policies,wrong_whole_stream_modulo_ticks=wrong_modulo,
                             wrong_whole_stream_modulo_indices=binary_lookup(timeline,wrong_modulo),
                             repeat_last_requested_not_last_video_frame=binary_lookup(timeline,short_tail),
                             final_video_frame_index=len(timeline.pts)-1),
                ordering=dict(unsorted_records=reordered_records,presentation_records=presentation,
                              presentation_labels=[label for _,label in presentation]),
                variants=variants,budget=dict(input=DEFAULT,**estimate_budget(DEFAULT)),
                latency=json.loads((HERE/"latency.json").read_text(encoding="utf-8")))


def outputs():
    from .render import render_outputs
    report = simulate()
    return {HERE/"report.json":json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--measure",action="store_true")
    args = parser.parse_args()
    if args.check and args.measure:
        parser.error("--check cannot replace sealed measurements")
    if args.measure:
        measure_latency()
    for path,content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 107 artifacts match." if args.check else "Wrote Round 107 artifacts.")


if __name__ == "__main__":
    main()
