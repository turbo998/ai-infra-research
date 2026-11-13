"""Original scalar RoPE layout contracts; no native FlashInfer/CUDA execution."""

import argparse
import cmath
import json
import math
from pathlib import Path
import platform
import statistics
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "a984f0f6e8c950571de8fbb12f5e0fd948ad1b0b"
MAX_EXACT = 2**53-1
DEFAULT = {"max_positions":131072,"rotary_dim":64,"head_dim":128,"cache_element_bytes":4,
           "cache_instances":1,"active_rows":4096,"query_heads":32,"key_heads":8,"qk_element_bytes":2}


def validate_vector(values,rotary_dim):
    values = tuple(values)
    if not values or type(rotary_dim) is not int or rotary_dim<2 or rotary_dim%2 or rotary_dim>len(values):
        raise ValueError("positive even rotary_dim no larger than head dimension required")
    if any(type(value) not in (int,float) or not math.isfinite(value) for value in values):
        raise ValueError("finite real vector required")
    return values


def frequencies(rotary_dim,theta=10000):
    if type(rotary_dim) is not int or rotary_dim<2 or rotary_dim%2:
        raise ValueError("positive even rotary dimension required")
    if type(theta) not in (int,float) or not math.isfinite(theta) or theta<=1:
        raise ValueError("finite original base theta greater than one required")
    return tuple(theta**(-2*i/rotary_dim) for i in range(rotary_dim//2))


def coefficients(position,rotary_dim,theta=10000):
    if type(position) is not int or abs(position)>MAX_EXACT:
        raise ValueError("bounded exact integer position required")
    angles = [position*frequency for frequency in frequencies(rotary_dim,theta)]
    return tuple(math.cos(angle) for angle in angles),tuple(math.sin(angle) for angle in angles)


def apply_coefficients(values,rotary_dim,cosines,sines,interleaved=True):
    values = validate_vector(values,rotary_dim)
    if type(interleaved) is not bool:
        raise ValueError("explicit Boolean layout required")
    if len(cosines)!=rotary_dim//2 or len(sines)!=rotary_dim//2:
        raise ValueError("coefficient half-width does not match rotary_dim")
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (*cosines,*sines)):
        raise ValueError("finite coefficients required")
    output = list(values)
    for pair,(cosine,sine) in enumerate(zip(cosines,sines)):
        first = 2*pair if interleaved else pair
        second = first+1 if interleaved else pair+rotary_dim//2
        a,b = values[first],values[second]
        output[first],output[second] = a*cosine-b*sine,a*sine+b*cosine
    return output


def rotate(values,position,rotary_dim,interleaved=True,theta=10000):
    values = validate_vector(values,rotary_dim)
    cosine,sine = coefficients(position,rotary_dim,theta)
    return apply_coefficients(values,rotary_dim,cosine,sine,interleaved)


def to_split(values,rotary_dim):
    values = validate_vector(values,rotary_dim)
    return list(values[:rotary_dim:2])+list(values[1:rotary_dim:2])+list(values[rotary_dim:])


def to_interleaved(values,rotary_dim):
    values = validate_vector(values,rotary_dim)
    half = rotary_dim//2
    prefix = [value for pair in zip(values[:half],values[half:rotary_dim]) for value in pair]
    return prefix+list(values[rotary_dim:])


def dot(left,right):
    if len(left)!=len(right):
        raise ValueError("matching vector widths required")
    return sum(a*b for a,b in zip(left,right))


def complex_relative_oracle(query,key,q_position,k_position,rotary_dim,theta=10000):
    query,key = validate_vector(query,rotary_dim),validate_vector(key,rotary_dim)
    if len(query)!=len(key):
        raise ValueError("matching query/key dimensions required")
    value = sum((complex(query[2*i],query[2*i+1]).conjugate()*
                 complex(key[2*i],key[2*i+1])*
                 cmath.exp(1j*(k_position-q_position)*frequency)).real
                for i,frequency in enumerate(frequencies(rotary_dim,theta)))
    return value+dot(query[rotary_dim:],key[rotary_dim:])


def dense_matrix(cosines,sines):
    width = 2*len(cosines)
    matrix = [[0.0]*width for _ in range(width)]
    for i,(cosine,sine) in enumerate(zip(cosines,sines)):
        matrix[2*i][2*i],matrix[2*i][2*i+1] = cosine,-sine
        matrix[2*i+1][2*i],matrix[2*i+1][2*i+1] = sine,cosine
    return matrix


def estimate_budget(p):
    for key in DEFAULT:
        value = p[key]
        minimum = 0 if key=="active_rows" else 1
        if type(value) is not int or not minimum<=value<=MAX_EXACT:
            raise ValueError(f"{key} requires a bounded exact integer >= {minimum}")
    if p["rotary_dim"]%2 or p["rotary_dim"]>p["head_dim"]:
        raise ValueError("rotary_dim must be even and no larger than head_dim")
    if p["cache_element_bytes"]!=4 or p["qk_element_bytes"] not in (2,4,8):
        raise ValueError("pinned cached API requires float32 coefficients; selected Q/K width must be two/four/eight")
    row_heads = p["active_rows"]*(p["query_heads"]+p["key_heads"])
    result = {"selected_cos_sin_cache_bytes":p["max_positions"]*p["rotary_dim"]*
              p["cache_element_bytes"]*p["cache_instances"],
              "selected_full_qk_payload_bytes":row_heads*p["head_dim"]*p["qk_element_bytes"],
              "selected_rotated_prefix_payload_bytes":row_heads*p["rotary_dim"]*p["qk_element_bytes"],
              "selected_untouched_tail_payload_bytes":row_heads*(p["head_dim"]-p["rotary_dim"])*p["qk_element_bytes"]}
    if any(value>MAX_EXACT for value in result.values()):
        raise ValueError("budget exceeds exact integer range")
    return {**result,"extra_allocation_bytes":None,"native_rope_latency_ms":None,
            "peak_hbm_bytes":None,"model_quality":None}


def measure_latency():
    rows = []
    for width in (8,16,32,64):
        values = [math.sin(i+.25) for i in range(width)]
        cosine,sine = coefficients(37,width)
        matrix = dense_matrix(cosine,sine)
        primary = lambda:apply_coefficients(values,width,cosine,sine)
        baseline = lambda:[sum(a*b for a,b in zip(row,values)) for row in matrix]
        error = max(abs(a-b) for a,b in zip(primary(),baseline()))
        assert error<1e-12
        for _ in range(2):
            primary()
            baseline()
        samples = {"primary":[],"baseline":[]}
        for repeat in range(7):
            order = (("primary",primary),("baseline",baseline))
            for name,operation in order if repeat%2==0 else reversed(order):
                start = time.perf_counter_ns()
                for _ in range(25):
                    operation()
                samples[name].append((time.perf_counter_ns()-start)/25e6)
        rows.append({"rotary_dim":width,"maximum_vector_error":error,
                     "primary_samples_ms":samples["primary"],"baseline_samples_ms":samples["baseline"],
                     "primary_median_ms":statistics.median(samples["primary"]),
                     "baseline_median_ms":statistics.median(samples["baseline"])})
    record = {"provenance":"measured","independently_reproduced":True,
              "implementation":"Original scalar pair rotations versus dense precomputed matrix application",
              "hardware":platform.processor() or platform.machine(),"python":platform.python_version(),
              "warmups":2,"repeats":7,"operations_per_sample":25,"order":"alternating",
              "exclusions":["coefficient computation","matrix construction","native FlashInfer","GPU memory/cache movement",
                            "kernel scheduling","attention computation"],"rows":rows}
    (HERE/"latency.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")


def simulate():
    query,key = [1,2,-1,.5,3,-2,7,-4],[.5,-1,2,1.5,-2,4,-3,6]
    rotary_dim,m,n = 6,3,9
    q_rot,k_rot = rotate(query,m,rotary_dim),rotate(key,n,rotary_dim)
    reference = complex_relative_oracle(query,key,m,n,rotary_dim)
    converted = dot(rotate(to_split(query,rotary_dim),m,rotary_dim,False),
                    rotate(to_split(key,rotary_dim),n,rotary_dim,False))
    wrong = dot(rotate(query,m,rotary_dim,False),rotate(key,n,rotary_dim,False))
    double_key = dot(q_rot,rotate(k_rot,n,rotary_dim))
    same = []
    for interleaved in (True,False):
        same.append({"interleaved":interleaved,"score":dot(rotate(query,m,rotary_dim,interleaved),
                                                          rotate(key,m,rotary_dim,interleaved)),
                     "unrotated_score":dot(query,key)})
    variants = []
    for shift in (0,7,128,4096):
        correct = dot(rotate(query,m+shift,rotary_dim),rotate(key,n+shift,rotary_dim))
        invalid = dot(rotate(query,m+shift,rotary_dim,False),rotate(key,n+shift,rotary_dim,False))
        variants.append({"shared_position_shift":shift,"correct_score":correct,
                         "correct_absolute_error":abs(correct-reference),"wrong_layout_score":invalid,
                         "wrong_layout_absolute_error":abs(invalid-reference),
                         "correct_relative_error_pct":100*abs(correct-reference)/abs(reference),
                         "wrong_relative_error_pct":100*abs(invalid-reference)/abs(reference)})
    return {"round":114,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"flashinfer-ai/flashinfer","pin":PIN,"date":"2026-11-10T23:25:15Z",
                      "license":"Apache-2.0","paths":["flashinfer/rope.py","include/flashinfer/pos_enc.cuh"]},
            "case":{"query":query,"key":key,"rotary_dim":rotary_dim,"head_dim":len(query),
                    "query_position":m,"key_position":n,"base_theta":10000,
                    "complex_relative_reference":reference,"interleaved_score":dot(q_rot,k_rot),
                    "converted_split_score":converted,"converted_absolute_error":abs(converted-reference),
                    "wrong_unconverted_layout_score":wrong,"wrong_layout_absolute_error":abs(wrong-reference),
                    "double_rotated_key_score":double_key,"double_rotation_absolute_error":abs(double_key-reference),
                    "tail_preserved":q_rot[rotary_dim:]==query[rotary_dim:] and k_rot[rotary_dim:]==key[rotary_dim:],
                    "same_position_controls":same,"tolerance":1e-10},
            "variants":variants,"budget":{"input":DEFAULT,**estimate_budget(DEFAULT)},
            "latency":json.loads((HERE/"latency.json").read_text(encoding="utf-8")),
            "unknowns":{"native_fix_reproduced":False,"gpu_latency_ms":None,"calibrated_confidence":None}}


def outputs():
    from experiments.rope_layout_partial_dimension_selection.render import render_outputs
    report = simulate()
    return {HERE/"report.json":json.dumps(report,indent=2,ensure_ascii=False)+"\n",**render_outputs(report)}


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
            if not path.exists() or path.read_text(encoding="utf-8")!=content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 114 artifacts match." if args.check else "Wrote Round 114 artifacts.")


if __name__=="__main__":
    main()
