"""Original logical 2:4 encoding, sparse product and amortization model."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path


SOURCE = "https://github.com/pytorch/pytorch/blob/5a7eae40861f6b8cc9ec71d3bc0e61f17a6568fa/torch/sparse/semi_structured.py"
WEIGHTS = [[4,3,0.2,-0.1,5,-4,0.3,0.2],
           [2,-5,0.4,0.1,3,2,-0.5,0.1],
           [-3,6,0.1,0.2,-2,5,0.3,0.4],
           [7,1,0.2,-0.4,4,-3,0.5,0.1]]
RHS = [[0.15*(i+1),(-1)**i*0.2,0.1*(i-2)] for i in range(8)]


def shape_of(matrix: list) -> tuple[int,int]:
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("matrix must be nonempty and rectangular")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("matrix values must be finite")
    return len(matrix),len(matrix[0])


def transpose(matrix: list) -> list:
    m,k = shape_of(matrix)
    return [[matrix[i][j] for i in range(m)] for j in range(k)]


def matmul(a: list,b: list) -> list:
    m,k = shape_of(a)
    bk,n = shape_of(b)
    if k != bk:
        raise ValueError("matrix multiplication inner dimensions must match")
    return [[math.fsum(a[i][p]*b[p][j] for p in range(k)) for j in range(n)] for i in range(m)]


def pattern_valid(matrix: list) -> bool:
    _,k = shape_of(matrix)
    if k % 4:
        return False
    return all(sum(v != 0 for v in row[start:start+4]) <= 2
               for row in matrix for start in range(0,k,4))


def prune_rows(matrix: list) -> list:
    _,k = shape_of(matrix)
    if k % 4:
        raise ValueError("row pruning needs complete adjacent quartets")
    result = []
    for row in matrix:
        output = [0.0]*k
        for start in range(0,k,4):
            keep = sorted(range(start,start+4),key=lambda j:(-abs(row[j]),j))[:2]
            for j in keep:
                output[j] = row[j]
        result.append(output)
    return result


def prune_bidirectional(matrix: list) -> list:
    m,k = shape_of(matrix)
    if m % 4 or k % 4:
        raise ValueError("bidirectional toy pruning needs complete 4x4 tiles")
    output = [[0.0]*k for _ in range(m)]
    for x in range(0,m,4):
        for y in range(0,k,4):
            rows,cols = [0]*4,[0]*4
            order = sorted(((i,j) for i in range(4) for j in range(4)),
                           key=lambda p:(-abs(matrix[x+p[0]][y+p[1]]),p[0],p[1]))
            for i,j in order:
                if rows[i] < 2 and cols[j] < 2:
                    output[x+i][y+j] = matrix[x+i][y+j]
                    rows[i] += 1
                    cols[j] += 1
    return output


def pack(matrix: list) -> dict:
    m,k = shape_of(matrix)
    if not pattern_valid(matrix):
        raise ValueError("logical pack requires at most two nonzeros per adjacent row quartet; prune explicitly")
    values,metadata = [],[]
    for row in matrix:
        packed,codes = [],[]
        for start in range(0,k,4):
            indices = [j for j in range(4) if row[start+j] != 0]
            indices += [j for j in range(4) if j not in indices][:2-len(indices)]
            indices.sort()
            packed.extend(row[start+j] for j in indices)
            codes.append(indices[0] | (indices[1] << 2))
        values.append(packed)
        metadata.append(codes)
    return {"shape":[m,k],"values":values,"logical_nibbles":metadata}


def unpack(encoded: dict, reverse_indices: bool = False) -> list:
    m,k = encoded["shape"]
    if type(m) is not int or type(k) is not int or m < 1 or k < 4 or k % 4:
        raise ValueError("invalid logical packed shape")
    if len(encoded["values"]) != m or len(encoded["logical_nibbles"]) != m:
        raise ValueError("packed row counts must match")
    output = [[0.0]*k for _ in range(m)]
    for i,(values,codes) in enumerate(zip(encoded["values"],encoded["logical_nibbles"])):
        if len(values) != k//2 or len(codes) != k//4 or any(not math.isfinite(v) for v in values):
            raise ValueError("packed values/metadata do not match shape")
        for q,code in enumerate(codes):
            if type(code) is not int or code not in (4,8,9,12,13,14):
                raise ValueError("invalid logical 2:4 nibble")
            indices = [code & 3,(code >> 2) & 3]
            if reverse_indices:
                indices.reverse()
            for j,index in enumerate(indices):
                output[i][4*q+index] = values[2*q+j]
    return output


def sparse_matmul(encoded: dict, rhs: list) -> list:
    m,k = encoded["shape"]
    # Validate the original format, but compute from packed values rather than dense rows.
    unpack(encoded)
    bk,n = shape_of(rhs)
    if k != bk:
        raise ValueError("packed product inner dimensions must match")
    result = []
    for values,codes in zip(encoded["values"],encoded["logical_nibbles"]):
        terms = [(4*q+(code & 3),values[2*q]) for q,code in enumerate(codes)]
        terms += [(4*q+((code >> 2) & 3),values[2*q+1]) for q,code in enumerate(codes)]
        result.append([math.fsum(value*rhs[index][j] for index,value in terms) for j in range(n)])
    if len(result) != m:
        raise ValueError("packed product row count mismatch")
    return result


def maximum_error(a: list,b: list) -> float:
    if shape_of(a) != shape_of(b):
        raise ValueError("numeric comparison needs matching shapes")
    return max(abs(x-y) for row,ref in zip(a,b) for x,y in zip(row,ref))


def align(value: int, multiple: int) -> int:
    return ((value+multiple-1)//multiple)*multiple


@dataclass(frozen=True)
class Budget:
    output_features: int = 4096
    input_features: int = 4096
    tokens: int = 128
    backend: str = "cutlass"
    reuse: int = 64
    pack_ms: float = 0.2
    dense_effective_tflops: float = 500.0
    sparse_effective_tflops: float = 800.0
    launch_us: float = 5.0


def estimate(b: Budget) -> dict:
    for name in ("output_features","input_features","tokens","reuse"):
        if type(getattr(b,name)) is not int or getattr(b,name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if b.backend not in ("cutlass","cusparselt"):
        raise ValueError("backend must be cutlass or cusparselt")
    for name in ("pack_ms","launch_us","dense_effective_tflops","sparse_effective_tflops"):
        value = getattr(b,name)
        if not math.isfinite(value) or value < 0 or (name.endswith("tflops") and value == 0):
            raise ValueError(f"{name} must be finite with a valid nonnegative rate/time")
    row_align,col_align = (32,64) if b.backend == "cutlass" else (16,16)
    m,k,n = b.output_features,b.input_features,b.tokens
    pm,pk,pn = align(m,row_align),align(k,col_align),align(n,8)
    tm,tk = align(k,row_align),align(m,col_align)
    values = pm*pk
    metadata = (pm*pk//4 + 1)//2
    compressed = values+metadata
    transpose_compressed = tm*tk+(tm*tk//4+1)//2
    dense_ms = 2*m*k*n/(b.dense_effective_tflops*1e9)+b.launch_us/1000
    sparse_ms = 2*pm*pk*pn/(b.sparse_effective_tflops*1e9)+b.launch_us/1000
    dense_total = b.reuse*dense_ms
    sparse_total = b.pack_ms+b.reuse*sparse_ms
    if not all(math.isfinite(value) for value in (dense_ms,sparse_ms,dense_total,sparse_total)):
        raise ValueError("modeled times must remain finite")
    saving = dense_ms-sparse_ms
    threshold = b.pack_ms/saving if saving > 0 else None
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("reuse threshold exceeds finite numerical range")
    minimum_reuse = math.floor(threshold)+1 if threshold is not None else None
    if minimum_reuse is not None:
        while b.pack_ms + minimum_reuse*sparse_ms >= minimum_reuse*dense_ms:
            minimum_reuse += 1
    return {
        "input":asdict(b),"scope_dtype":"FP16/BF16 only; two-byte values, four logical metadata bits per quartet",
        "forward_padded_shape":[pm,pk],"transpose_padded_shape":[tm,tk],"padded_tokens":pn,
        "source_shape_gate_passes_without_padding":m % row_align == 0 and k % col_align == 0,
        "original_dense_weight_bytes":2*m*k,"padded_dense_weight_bytes":2*pm*pk,
        "retained_value_bytes":values,"logical_metadata_bytes":metadata,"logical_compressed_bytes":compressed,
        "both_orientation_logical_bytes":compressed+transpose_compressed,
        "compressed_to_original_ratio":compressed/(2*m*k),
        "dense_modeled_kernel_ms":dense_ms,"sparse_modeled_kernel_ms":sparse_ms,
        "minimum_reuse_for_strict_saving":minimum_reuse,
        "dense_modeled_total_ms":dense_total,"sparse_modeled_total_ms":sparse_total,
        "gpu_latency_ms":None,"model_accuracy":None,"actual_backend_packed_bytes":None,
        "scope":"original logical metadata only; no backend swizzle, mask, optimizer or activation storage",
    }


def simulate() -> dict:
    row_pruned = prune_rows(WEIGHTS)
    encoded = pack(row_pruned)
    reference = matmul(row_pruned,RHS)
    original = matmul(WEIGHTS,RHS)
    bidirectional = prune_bidirectional(WEIGHTS)
    both_output = sparse_matmul(pack(bidirectional),RHS)
    cases = [
        {"id":"logical-packed-product","maximum_absolute_error":maximum_error(sparse_matmul(encoded,RHS),reference),
         "reference":"same-row-pruned-dense","intentional_pruning":False},
        {"id":"reverse-metadata-index-order","maximum_absolute_error":maximum_error(
            matmul(unpack(encoded,True),RHS),reference),"reference":"same-row-pruned-dense","intentional_pruning":False},
        {"id":"pruned-vs-original-dense","maximum_absolute_error":maximum_error(reference,original),
         "reference":"original-unpruned-dense","intentional_pruning":True},
        {"id":"bidirectional-pruned-product","maximum_absolute_error":maximum_error(
            both_output,matmul(bidirectional,RHS)),"reference":"same-bidirectionally-pruned-dense","intentional_pruning":False},
        {"id":"bidirectional-vs-row-pruning","maximum_absolute_error":maximum_error(both_output,reference),
         "reference":"row-pruned-dense","intentional_pruning":True},
    ]
    for row in cases:
        row["matches_reference"] = row["maximum_absolute_error"] < 1e-12
        row["confidence"] = None
    b = Budget()
    return {
        "schema_version":1,"round":100,"provenance":"synthetic","independently_reproduced":False,
        "physical_gpu_measurement":False,"source":SOURCE,
        "fixture":{"weights":WEIGHTS,"rhs":RHS,"row_pruned":row_pruned,"encoded":encoded,
                   "row_pruned_reference":reference,"bidirectional_pruned":bidirectional,"tolerance":1e-12},
        "cases":cases,
        "transpose_checks":{
            "row_pruned_forward_valid":pattern_valid(row_pruned),
            "row_pruned_transpose_valid":pattern_valid(transpose(row_pruned)),
            "bidirectional_forward_valid":pattern_valid(bidirectional),
            "bidirectional_transpose_valid":pattern_valid(transpose(bidirectional)),
        },
        "budget":estimate(b),
        "scaling":[estimate(replace(b,tokens=n)) for n in (1,8,128,1024)],
        "notes":[
            "Original logical nibble layout, not CUTLASS/cuSPARSELt swizzled storage or GPU execution.",
            "Compression is not pruning; numerical parity is against the same pruned dense matrix.",
            "Row-valid 2:4 sparsity need not remain valid under transpose; another orientation can need another mask.",
            "Both-orientation toy pruning is an original greedy capacity-constrained policy, not source algorithm reproduction.",
            "Pinned FP16/BF16 shape gates differ by backend; caller-side padding is explicit, not automatic sparse conversion.",
            "FP32 CUTLASS uses a different 1:2 conversion branch and FP8 needs separate APIs/backend gates; neither is modeled.",
            "Rates, launch and packing times are assumptions; effective TFLOPS counts original-equivalent dense operations.",
        ],
    }


def outputs() -> dict[Path,str]:
    from .render import render_outputs
    report = simulate()
    return {Path(__file__).with_name("report.json"):json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    for path,content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 100 artifacts match." if args.check else "Wrote Round 100 artifacts.")


if __name__ == "__main__":
    main()
