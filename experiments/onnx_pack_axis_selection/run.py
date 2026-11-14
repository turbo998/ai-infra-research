"""Original rank-two UINT8-code packing model;not ONNX/ORT execution."""

import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="69bf957565843052ffb46e0dc121c38456741288"
DEFAULT={"shape":[2,3],"values":[0,1,7,2,3,4],"axis":1,"bits":3,"radix":None,"group_size":None,"word_bits":8}


def integer(value,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError(f"exact integer in {low}..{high} required")
    return value


def parameters(p):
    bits=p["bits"];radix=p["radix"];group=p["group_size"]
    if bits is not None:
        integer(bits,1,8)
        if radix is not None or group is not None:
            raise ValueError("bits excludes radix/group_size")
        radix=1<<bits;group=1
    else:
        integer(radix,2,256);integer(group,1,62)
    limit=radix**group
    if limit>2**63-1:
        raise ValueError("radix power must fit positive INT64")
    return radix,group,(limit-1).bit_length(),limit


def validate(p):
    if not isinstance(p,dict) or set(p)!={"shape","values","axis","bits","radix","group_size","word_bits"}:
        raise ValueError("exact original rank-two code tensor and packing metadata required")
    if not isinstance(p["shape"],list) or len(p["shape"])!=2:
        raise ValueError("original rank-two shape required")
    for dim in p["shape"]:
        integer(dim,0,16)
    if p["shape"][0]*p["shape"][1]>128 or not isinstance(p["values"],list) or len(p["values"])!=p["shape"][0]*p["shape"][1]:
        raise ValueError("matched row-major values with at most128codes required")
    integer(p["axis"],-2,1)
    if type(p["word_bits"]) is not int or p["word_bits"] not in (8,32):
        raise ValueError("UINT8 or UINT32 storage width required")
    params=parameters(p)
    for value in p["values"]:
        integer(value,0,params[0]-1)
    return params


def words_required(count,group,width,word_bits):
    groups=(count+group-1)//group
    return (groups*width+word_bits-1)//word_bits


def lanes(shape,values,axis):
    rows,cols=shape
    return [[values[r*cols+c] for c in range(cols)] for r in range(rows)] if axis==1 else [
        [values[r*cols+c] for r in range(rows)] for c in range(cols)]


def assemble(lines,shape,axis):
    values=[0]*(shape[0]*shape[1])
    for other,line in enumerate(lines):
        for index,value in enumerate(line):
            values[other*shape[1]+index if axis==1 else index*shape[1]+other]=value
    return values


def group_codes(values,radix,group):
    for start in range(0,len(values),group):
        code=0
        for digit in reversed(values[start:start+group]):
            code=code*radix+digit
        yield code


def pack_lane(values,radix,group,width,word_bits):
    reservoir=0;held=0;mask=(1<<word_bits)-1;words=[]
    for code in group_codes(values,radix,group):
        reservoir|=code<<held;held+=width
        while held>=word_bits:
            words.append(reservoir&mask);reservoir>>=word_bits;held-=word_bits
    if held:
        words.append(reservoir)
    return words


def unpack_lane(words,count,radix,group,width,limit,word_bits):
    reservoir=0;held=0;cursor=0;values=[];mask=(1<<width)-1
    for _ in range((count+group-1)//group):
        while held<width:
            reservoir|=words[cursor]<<held;held+=word_bits;cursor+=1
        code=reservoir&mask;reservoir>>=width;held-=width
        if code>=limit:
            raise ValueError("original strict validator rejects unused radix group index")
        for _ in range(min(group,count-len(values))):
            values.append(code%radix);code//=radix
    return values


def pack(p):
    radix,group,width,limit=validate(p);axis=p["axis"]%2;shape=list(p["shape"])
    shape[axis]=words_required(shape[axis],group,width,p["word_bits"])
    lines=[pack_lane(line,radix,group,width,p["word_bits"]) for line in lanes(p["shape"],p["values"],axis)]
    values=assemble(lines,shape,axis)
    return {"shape":shape,"words":values,"word_bits":p["word_bits"],
            "raw_little_endian_bytes":[byte for value in values for byte in value.to_bytes(p["word_bits"]//8,"little")],
            "parameters":{"radix":radix,"group_size":group,"group_width":width,"radix_power_decimal":str(limit)},
            "selected_storage_bytes":len(values)*(p["word_bits"]//8)}


def unpack(packet,p,count):
    radix,group,width,limit=validate(p);axis=p["axis"]%2;integer(count,0,16)
    if not isinstance(packet,dict) or not {"shape","words","word_bits"}<=set(packet):
        raise ValueError("original packed shape,word list and storage width required")
    if packet["word_bits"]!=p["word_bits"] or type(packet["word_bits"]) is not int:
        raise ValueError("packed storage width disagrees with declared metadata")
    shape=packet["shape"]
    if not isinstance(shape,list) or len(shape)!=2 or any(type(v) is not int or not 0<=v<=128 for v in shape):
        raise ValueError("original packed rank-two shape required")
    if shape[1-axis]!=p["shape"][1-axis] or shape[axis]!=words_required(count,group,width,p["word_bits"]):
        raise ValueError("packed selected axis must have exact words required by count;other axis preserved")
    if not isinstance(packet["words"],list) or len(packet["words"])!=shape[0]*shape[1]:
        raise ValueError("matched packed row-major words required")
    for word in packet["words"]:
        integer(word,0,(1<<p["word_bits"])-1)
    output_shape=list(shape);output_shape[axis]=count
    if output_shape[0]*output_shape[1]>128:
        raise ValueError("original UI decoded code bound exceeded")
    lines=[unpack_lane(line,count,radix,group,width,limit,p["word_bits"]) for line in lanes(shape,packet["words"],axis)]
    return {"shape":output_shape,"values":assemble(lines,output_shape,axis)}


def bitwise_pack_lane(values,radix,group,width,word_bits):
    words=[0]*words_required(len(values),group,width,word_bits);cursor=0
    for code in group_codes(values,radix,group):
        for bit in range(width):
            words[cursor//word_bits]|=((code>>bit)&1)<<(cursor%word_bits);cursor+=1
    return words


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        values=[i%8 for i in range(n)]
        funcs=(lambda:pack_lane(values,8,1,3,8),lambda:bitwise_pack_lane(values,8,1,3,8))
        assert funcs[0]()==funcs[1]()
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
        rows.append({"codes":n,"reservoir_samples_ms":samples[0],"bitwise_samples_ms":samples[1],
                     "reservoir_median_ms":statistics.median(samples[0]),"bitwise_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original single-lane3-bit packing:bounded reservoir versus individual bit writes;both include code grouping and output construction",
            "excludes":"input/parameter preparation,matrix validation,ONNX/reference/function/ORT/backend evaluation,GPU and network;timer sizes exceed UI bound","rows":rows}


def build_report(latency):
    packet=pack(DEFAULT);decoded=unpack(packet,DEFAULT,3)
    axis0={**copy.deepcopy(DEFAULT),"axis":0}
    flat={**copy.deepcopy(DEFAULT),"shape":[1,6]}
    ternary={**copy.deepcopy(DEFAULT),"shape":[1,5],"values":[2,1,0,2,1],"bits":None,"radix":3,"group_size":5}
    two_bits={**copy.deepcopy(ternary),"bits":2,"radix":None,"group_size":None}
    u32={**copy.deepcopy(DEFAULT),"shape":[1,7],"values":[31]*7,"bits":5,"word_bits":32}
    padding=copy.deepcopy(packet);padding["words"][1]|=128
    invalid=pack(ternary);invalid["words"][0]=255
    try:
        unpack(invalid,ternary,5)
    except ValueError as error:
        strict_error=str(error)
    else:
        raise RuntimeError("original strict group-index rejection missing")
    return {"schema_version":1,"round":147,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"onnx/onnx","pin":PIN,"license":"Apache-2.0 source rights preserved;no implementation mirrored",
                      "verified":"new opset29Pack/Unpack,UINT8 codes to UINT8/UINT32 words,LSB-first per-axis lanes,mandatory scalar INT64count,radix-group width and padding,invalid inputs unspecified",
                      "chronology":"Nov12UTC22:21 = Nov13BJT06:21feature addition,not a Nov14commit;recent capability reviewed on Nov14",
                      "scope":"original strict rank-two integer-code model;no numerical quantization,INT4dtype conversion,native ONNX/reference/function/ORT,GPU execution or invalid-input guarantee"},
            "case":{"inputs":DEFAULT,"packed":packet,"decoded":decoded,"axis0_packed":pack(axis0),"flattened_other_tensor":pack(flat),
                    "same_word_count_wrong_count":unpack(packet,DEFAULT,4),"padding_flipped_decoded":unpack(padding,DEFAULT,3),
                    "padding_flipped_words":padding["words"]},
            "ternary":{"inputs":ternary,"packed":pack(ternary),"two_bit_alternative":pack(two_bits),
                       "unused_index_strict_error":strict_error,"unused_indices":256-3**5},
            "uint32":{"inputs":u32,"packed":pack(u32),"uint8_alternative":pack({**u32,"word_bits":8}),
                      "decoded":unpack(pack(u32),u32,7)},"latency":latency,
            "native_onnx_evaluation_verified":None,"native_function_lowering_verified":None,"onnxruntime_backend_support_verified":None,
            "gpu_latency_ms":None,"native_runtime_peak_memory_bytes":None}


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
    print("Round147 sealed artifacts match" if args.check else "Wrote Round147 original artifacts")


if __name__=="__main__":
    main()
