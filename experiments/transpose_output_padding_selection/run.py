"""Original finite-domain scalar transpose scatter versus a dense forward-matrix oracle."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="7bbc31dd45f4e43a5e3899e7e328aedfd967f2ff"
PARAMS={"stride":2,"padding":1,"dilation":1,"output_padding":1}
DEFAULT={"batch":8,"in_channels":64,"out_channels":32,"groups":4,"input_length":128,"kernel_size":3,
         **PARAMS,"scalar_bytes":4}


def output_length(length,kernel_size,p):
    if set(p)!=set(PARAMS) or type(length) is not int or type(kernel_size) is not int or not 1<=length<=1024 or not 1<=kernel_size<=31:
        raise ValueError("bounded positive exact input/kernel lengths and parameter keys required")
    if any(type(p[k]) is not int for k in p) or not 1<=p["stride"]<=16 or not 1<=p["dilation"]<=16 or not 0<=p["padding"]<=128 or not 0<=p["output_padding"]<p["stride"]:
        raise ValueError("bounded stride/dilation/padding;original output_padding must be below stride")
    result=(length-1)*p["stride"]-2*p["padding"]+p["dilation"]*(kernel_size-1)+p["output_padding"]+1
    if not 1<=result<=32768:
        raise ValueError("selected positive output length must be at most32768")
    return result


def validate_values(values):
    if any(type(v) is not int or abs(v)>1024 for v in values):
        raise ValueError("exact scalar integers with absolute value at most1024 required")


def transpose(x,kernel,p):
    length=output_length(len(x),len(kernel),p);validate_values(x);validate_values(kernel)
    result=[0]*length
    for i,value in enumerate(x):
        for tap,weight in enumerate(kernel):
            position=i*p["stride"]-p["padding"]+tap*p["dilation"]
            if 0<=position<length:
                result[position]+=value*weight
    return result


def matrix_oracle(input_length,kernel,p):
    columns=output_length(input_length,len(kernel),p)
    rows=(columns+2*p["padding"]-p["dilation"]*(len(kernel)-1)-1)//p["stride"]+1
    if rows!=input_length:
        raise ValueError("finite forward matrix does not match selected transpose input")
    result=[]
    for row in range(rows):
        values=[]
        for col in range(columns):
            delta=col+p["padding"]-row*p["stride"]
            k=delta//p["dilation"]
            values.append(kernel[k] if delta>=0 and delta%p["dilation"]==0 and k<len(kernel) else 0)
        result.append(values)
    return result


def matrix_transpose_apply(matrix,x):
    return [sum(matrix[i][j]*x[i] for i in range(len(x))) for j in range(len(matrix[0]))]


def requested_output_padding(length,kernel_size,p,target):
    base=output_length(length,kernel_size,p)-p["output_padding"]
    if type(target) is not int or target<1 or not base<=target<base+p["stride"]:
        raise ValueError("requested output length outside selected stride-ambiguity interval")
    return target-base


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int for v in p.values()) or any(not 1<=p[k]<=65536 for k in ("batch","in_channels","out_channels","groups")) or p["scalar_bytes"] not in (1,2,4,8):
        raise ValueError("bounded exact grouped payload parameters required")
    if p["in_channels"]%p["groups"] or p["out_channels"]%p["groups"]:
        raise ValueError("channels must divide evenly by groups")
    out=output_length(p["input_length"],p["kernel_size"],{k:p[k] for k in PARAMS})
    inp=p["batch"]*p["in_channels"]*p["input_length"]*p["scalar_bytes"]
    output=p["batch"]*p["out_channels"]*out*p["scalar_bytes"]
    weights=p["in_channels"]*(p["out_channels"]//p["groups"])*p["kernel_size"]*p["scalar_bytes"]
    if max(inp,output,weights,inp+output+weights)>2**53-1:
        raise ValueError("payload exceeds shared exact integer budget")
    return {"output_length":out,"input_payload_bytes":inp,"output_payload_bytes":output,"weight_payload_bytes":weights,
            "selected_read_input_and_weights_once_write_output_bytes":inp+weights+output,
            "note":"grouped logical tensor payload only;scalar evaluator is onechannel;no native traffic,workspace,allocator or convolution algorithm"}


def timing_paths(n):
    x=[i%5-2 for i in range(n)];kernel=[1,2,3];matrix=matrix_oracle(n,kernel,PARAMS)
    def scatter():
        return transpose(x,kernel,PARAMS)
    def dense():
        return matrix_transpose_apply(matrix,x)
    return scatter,dense


def measure():
    rows=[]
    for n in (16,32,64,128):
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
        rows.append({"input_length":n,"scatter_samples_ms":samples[0],"dense_samples_ms":samples[1],
                     "scatter_median_ms":statistics.median(samples[0]),"dense_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"original Python scalar tap scatter versus prebuilt dense forward-matrix transpose application",
            "excludes":"dense matrix construction,PyTorch/native BLAS/cuDNN,GPU,bias,groups evaluation,streaming",
            "baseline":"educational zero-filled matrix,not an actual native convolution algorithm","rows":rows}


def build_report(latency):
    x=[1,2];kernel=[1,2,3];base=transpose(x,kernel,{**PARAMS,"output_padding":0});extended=transpose(x,kernel,PARAMS)
    matrix=matrix_oracle(len(x),kernel,PARAMS);z=[1,2,3,4]
    az=[sum(a*b for a,b in zip(row,z)) for row in matrix]
    return {"schema_version":1,"round":139,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"pytorch/pytorch","pin":PIN,"license":"BSD-style",
                      "verified":"ConvTranspose1d input-gradient/notinverse,output-lengthformula,output_padding selects shape not zeroappend,weight layout and output_size stride interval",
                      "chronology":"conv.py lasttrackedJul20doc/commenttypos;Nov14regionalAOTItreecontext is not new ConvTranspose",
                      "model_policy":"singlechannel integer/no bias;output_padding<stride subset for exact finite forward-adjoint dimensions,not all native dilation-related cases"},
            "case":{"input_values":x,"kernel":kernel,"parameters":PARAMS,
                    "output_padding0":base,"output_padding1":extended,"wrong_zero_appended_output":base+[0],
                    "dense_forward_matrix_for_length4":matrix,"oracle_transpose":matrix_transpose_apply(matrix,x),
                    "forward_input":z,"forward_output":az,"transpose_after_forward":transpose(az,kernel,PARAMS),
                    "adjoint_left":sum(a*b for a,b in zip(az,x)),"adjoint_right":sum(a*b for a,b in zip(z,extended)),
                    "requested_length4_output_padding":requested_output_padding(len(x),len(kernel),PARAMS,4),
                    "output_padding1_matching_wrong_append_positions":sum(a==b for a,b in zip(extended,base+[0])),
                    "output_position_denominator":len(extended)},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_pytorch_parity":None,"native_gpu_latency_ms":None,"native_workspace_bytes":None,"audio_image_quality":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round139 sealed artifacts match" if a.check else "Wrote Round139 original artifacts")


if __name__=="__main__":
    main()
