"""Original exact CTC path/prefix references and sealed local CPU timings."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import time


HERE = Path(__file__).resolve().parent
SOURCE = "https://github.com/kensho-technologies/pyctcdecode/blob/afecb67622c1395b85b6a55d2902a780c0c927d4/pyctcdecode/decoder.py"
NEG_INF = -math.inf
FRAMES = [[0.45,0.35,0.2],[0.45,0.35,0.2]]
REPEATS = [[0.1,0.8,0.1],[0.6,0.2,0.2],[0.1,0.8,0.1]]
STREAM = [[0.1,0.8,0.1],[0.1,0.8,0.1]]


def validate_frames(frames: list):
    if not frames:
        return
    width = len(frames[0])
    if width < 2 or any(len(row) != width for row in frames):
        raise ValueError("frames need a blank and at least one label with matching widths")
    if any(not math.isfinite(p) or p < 0 for row in frames for p in row):
        raise ValueError("frame probabilities must be finite and nonnegative")
    if any(not math.isclose(math.fsum(row),1.0,abs_tol=1e-12,rel_tol=0) for row in frames):
        raise ValueError("each probability frame must sum to one")


def logadd(a: float,b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    maximum = max(a,b)
    return maximum+math.log1p(math.exp(-abs(a-b)))


def log_probability(p: float) -> float:
    return math.log(p) if p else NEG_INF


def collapse(path: tuple, remove_blank_first: bool = False) -> tuple:
    if remove_blank_first:
        path = tuple(token for token in path if token != 0)
    result = []
    previous = None
    for token in path:
        if token != 0 and token != previous:
            result.append(token)
        previous = token
    return tuple(result)


def enumerate_paths(frames: list, use_max: bool = False, remove_blank_first: bool = False) -> dict:
    validate_frames(frames)
    if len(frames) > 12:
        raise ValueError("exhaustive reference is capped at twelve frames")
    if not frames:
        return {():1.0}
    result = {}
    for path in itertools.product(range(len(frames[0])),repeat=len(frames)):
        probability = math.prod(row[token] for row,token in zip(frames,path))
        transcript = collapse(path,remove_blank_first)
        result[transcript] = (max(result.get(transcript,0.0),probability) if use_max else
                              result.get(transcript,0.0)+probability)
    return result


def prefix_decode(frames: list, state: dict | None = None, beam_width: int | None = None,
                  merge_repeat_ends: bool = False) -> dict:
    validate_frames(frames)
    if beam_width is not None and (type(beam_width) is not int or beam_width < 1):
        raise ValueError("beam width must be a positive integer or None for exact decoding")
    current = {():(0.0,NEG_INF)} if state is None else dict(state)
    if not current:
        raise ValueError("streaming state must contain at least one prefix")
    for row in frames:
        following = {}

        def add(prefix,which,value):
            if value == NEG_INF:
                return
            pair = following.setdefault(prefix,[NEG_INF,NEG_INF])
            pair[which] = logadd(pair[which],value)

        for prefix,(blank,nonblank) in current.items():
            total = logadd(blank,nonblank)
            add(prefix,0,total+log_probability(row[0]))
            for token in range(1,len(row)):
                lp = log_probability(row[token])
                if prefix and prefix[-1] == token:
                    add(prefix,1,nonblank+lp)
                    add(prefix+(token,),1,(total if merge_repeat_ends else blank)+lp)
                else:
                    add(prefix+(token,),1,total+lp)
        current = {prefix:tuple(values) for prefix,values in following.items()}
        if beam_width is not None:
            current = dict(sorted(current.items(),key=lambda item:(-logadd(*item[1]),item[0]))[:beam_width])
    return current


def probabilities(state: dict) -> dict:
    return {prefix:math.exp(logadd(*pair)) for prefix,pair in state.items()}


def target_log_probability(frames: list,target: tuple[int,...]) -> float:
    """Independent fixed-target forward automaton, with no free-prefix expansion."""
    validate_frames(frames)
    if any(type(token) is not int or token < 1 or (frames and token >= len(frames[0])) for token in target):
        raise ValueError("target labels must exclude blank and fit the alphabet")
    expanded = [0]
    for token in target:
        expanded += [token,0]
    state = [0.0]+[NEG_INF]*(len(expanded)-1)
    for row in frames:
        next_state = []
        for s,token in enumerate(expanded):
            value = state[s]
            if s >= 1:
                value = logadd(value,state[s-1])
            if s >= 2 and token != 0 and token != expanded[s-2]:
                value = logadd(value,state[s-2])
            next_state.append(value+log_probability(row[token]))
        state = next_state
    return state[0] if not target else logadd(state[-1],state[-2])


def concat_chunk_distributions(chunks: list) -> dict:
    result = {():1.0}
    for chunk in chunks:
        local = enumerate_paths(chunk)
        merged = {}
        for left,p in result.items():
            for right,q in local.items():
                key = left+right
                merged[key] = merged.get(key,0.0)+p*q
        result = merged
    return result


def decode_chunks(frames: list,sizes: list[int]) -> dict:
    if not sizes or any(type(size) is not int or size < 1 for size in sizes) or sum(sizes) != len(frames):
        raise ValueError("chunk sizes must cover all frames exactly")
    state = None
    start = 0
    for size in sizes:
        state = prefix_decode(frames[start:start+size],state)
        start += size
    return probabilities(state)


def difference(a: dict,b: dict) -> float:
    return max(abs(a.get(prefix,0.0)-b.get(prefix,0.0)) for prefix in a.keys() | b.keys())


def transcript(prefix: tuple) -> str:
    return "".join(chr(96+token) for token in prefix)


def distribution_rows(values: dict) -> list:
    return [{"transcript":transcript(prefix),"probability":value}
            for prefix,value in sorted(values.items(),key=lambda item:(-item[1],item[0]))]


def timing_fixture(count: int) -> list:
    result = []
    for i in range(count):
        weights = [1+i%3,2+(2*i)%5,1+(3*i)%4]
        result.append([w/sum(weights) for w in weights])
    return result


def measure_latency():
    rows = []
    for count in (3,5,7,9):
        frames = timing_fixture(count)
        paths = {
            "prefix_dp":lambda:probabilities(prefix_decode(frames)),
            "path_enumeration":lambda:enumerate_paths(frames),
        }
        if difference(paths["prefix_dp"](),paths["path_enumeration"]()) > 1e-12:
            raise ValueError("timed exact implementations disagree")
        for function in paths.values():
            for _ in range(2):
                function()
        samples = {key:[] for key in paths}
        for repeat in range(7):
            for key in (list(paths) if repeat % 2 == 0 else list(reversed(paths))):
                start = time.perf_counter_ns()
                paths[key]()
                samples[key].append(time.perf_counter_ns()-start)
        rows.append({"frames":count,"samples_ns":samples,
                     **{key+"_median_ms":statistics.median(value)/1e6 for key,value in samples.items()}})
    data = {
        "provenance":"measured","measured_at_utc":datetime.now(timezone.utc).isoformat(),
        "environment":{"os":platform.system(),"machine":platform.machine(),
                       "processor":platform.processor(),"python":platform.python_version()},
        "scope":"original standard-library exact CTC functions including validation; no model, LM, native library or GPU",
        "fixture":{"labels_without_blank":2,"weights":"[1+i%3, 2+(2*i)%5, 1+(3*i)%4], divided by their sum"},
        "timer":"perf_counter_ns","warmups_per_path":2,"repeats_per_path":7,
        "order":"alternating function order by repeat","rows":rows,
    }
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8",newline="\n")


def load_latency() -> dict:
    path = HERE/"latency.json"
    if not path.exists():
        raise ValueError("missing sealed CPU timings; run with --measure before rendering")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["provenance"] != "measured" or data["repeats_per_path"] != 7:
        raise ValueError("invalid measured timing contract")
    if [row["frames"] for row in data["rows"]] != [3,5,7,9]:
        raise ValueError("sealed timing tiers do not match the fixture")
    for row in data["rows"]:
        for key in ("prefix_dp","path_enumeration"):
            samples = row["samples_ns"][key]
            if len(samples) != 7 or any(type(n) is not int or n <= 0 for n in samples):
                raise ValueError("timing samples must be seven positive nanosecond integers")
            if row[key+"_median_ms"] != statistics.median(samples)/1e6:
                raise ValueError("sealed timing median does not match samples")
    return data


@dataclass(frozen=True)
class Budget:
    frames: int = 100
    labels: int = 32
    beam_width: int = 64
    token_index_bytes: int = 4
    frame_ms: float = 20.0


def estimate(b: Budget) -> dict:
    for name in ("frames","labels","beam_width","token_index_bytes"):
        if type(getattr(b,name)) is not int or not 1 <= getattr(b,name) <= 9007199254740991:
            raise ValueError(f"{name} must be a positive safe integer")
    if b.token_index_bytes not in (2,4,8) or type(b.frame_ms) not in (int,float) or not math.isfinite(b.frame_ms) or b.frame_ms <= 0:
        raise ValueError("index bytes must be 2/4/8 and frame duration finite positive")
    if b.labels >= 1 << (8*b.token_index_bytes):
        raise ValueError("labels exceed the selected unsigned index capacity")
    counters = (b.beam_width*(b.labels+2),b.beam_width*16,b.beam_width*b.frames*b.token_index_bytes)
    if any(n > 9007199254740991 for n in counters):
        raise ValueError("payload bound exceeds exact numerical range")
    log_paths = b.frames*math.log10(b.labels+1)
    exact_paths = 1
    for _ in range(b.frames):
        if exact_paths > 9007199254740991//(b.labels+1):
            exact_paths = None
            break
        exact_paths *= b.labels+1
    audio_span = b.frames*b.frame_ms
    if not math.isfinite(log_paths) or not math.isfinite(audio_span):
        raise ValueError("modeled span/path logarithm exceeds finite numerical range")
    return {
        "input":asdict(b),"path_count_if_js_safe":exact_paths,"log10_unmerged_path_count":log_paths,
        "per_frame_contribution_bound":b.beam_width*(b.labels+2),
        "two_log_scores_bytes":b.beam_width*2*8,
        "prefix_token_payload_upper_bytes":b.beam_width*b.frames*b.token_index_bytes,
        "audio_span_ms":audio_span,
        "gpu_latency_ms":None,"asr_wer":None,"calibrated_confidence":None,
        "scope":"pruned beam cap and logical payload only; no Python objects, LM state, timing metadata or model inference",
    }


def simulate() -> dict:
    reference = enumerate_paths(FRAMES)
    repeated = enumerate_paths(REPEATS)
    stream_ref = enumerate_paths(STREAM)
    state = prefix_decode(STREAM[:1])
    carried = probabilities(prefix_decode(STREAM[1:],state))
    cases = []

    def case(name,actual,expected,fixture,policy=False):
        error = difference(actual,expected)
        cases.append({
            "id":name,"fixture":fixture,"scores":distribution_rows(actual),
            "maximum_probability_or_score_error":error,"matches_reference":error < 1e-12,
            "intentional_search_policy":policy,"total_score_mass":math.fsum(actual.values()),"confidence":None,
        })

    case("exact-prefix-logsum",probabilities(prefix_decode(FRAMES)),reference,"greedy-counterexample")
    case("best-path-per-transcript",enumerate_paths(FRAMES,use_max=True),reference,"greedy-counterexample",True)
    case("remove-blank-before-repeat-collapse",enumerate_paths(REPEATS,remove_blank_first=True),
         repeated,"separated-repeat")
    case("merge-blank-nonblank-repeat-ends",probabilities(prefix_decode(STREAM,merge_repeat_ends=True)),
         stream_ref,"streaming-repeat")
    case("carried-prefix-state",carried,stream_ref,"streaming-repeat")
    case("reset-and-concatenate-chunks",concat_chunk_distributions([STREAM[:1],STREAM[1:]]),
         stream_ref,"streaming-repeat")
    case("beam-one-pruning",probabilities(prefix_decode(FRAMES,beam_width=1)),
         reference,"greedy-counterexample",True)
    winner = max(reference,key=reference.get)
    greedy_path = tuple(max(range(len(row)),key=row.__getitem__) for row in FRAMES)
    long_frames = [[0.5,0.5] for _ in range(2000)]
    target_log = target_log_probability(long_frames,(1,))
    return {
        "schema_version":1,"round":101,"provenance":"synthetic","independently_reproduced":False,
        "physical_gpu_measurement":False,"source":SOURCE,
        "fixtures":{"greedy-counterexample":FRAMES,"separated-repeat":REPEATS,"streaming-repeat":STREAM},
        "greedy":{"path":greedy_path,"transcript":transcript(collapse(greedy_path)),
                  "path_probability":math.prod(row[token] for row,token in zip(FRAMES,greedy_path))},
        "best_transcript":{"text":transcript(winner),"probability":reference[winner]},
        "cases":cases,
        "partition_checks":[{"sizes":sizes,"maximum_error":difference(decode_chunks(REPEATS,sizes),repeated)}
                            for sizes in ([3],[1,2],[2,1],[1,1,1])],
        "underflow":{"frames":2000,"target":"a","linear_probability":2000*2001/2*(0.5**2000),
                     "log_probability":target_log,
                     "closed_form_log_probability":math.log(2000*2001/2)-2000*math.log(2),
                     "note":"zero is float underflow of a positive event, not impossible transcript or confidence"},
        "budget":estimate(Budget()),"latency":load_latency(),"asr_quality":None,
        "notes":[
            "Pure acoustic CTC only: no language model, BPE, hotwords, timestamps or calibrated confidence.",
            "Exact prefix DP retains all prefixes; a finite beam is a different approximate search policy.",
            "Best path per transcript is Viterbi scoring, not summed transcript posterior.",
            "Streaming carries blank/nonblank ends; resetting and concatenating can invent impossible repeats.",
            "Measured CPU timings compare original exact functions, not pyctcdecode/NeMo or production speech latency.",
        ],
    }


def outputs() -> dict[Path,str]:
    from .render import render_outputs
    report = simulate()
    return {HERE/"report.json":json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--measure",action="store_true",help="explicitly replace sealed local CPU samples")
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
    print("Round 101 artifacts match." if args.check else "Wrote Round 101 artifacts.")


if __name__ == "__main__":
    main()
