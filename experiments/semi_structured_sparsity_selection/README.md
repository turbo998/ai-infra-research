# Logical 2:4 sparse GEMM / 半结构化稀疏矩阵契约

Round 100 separates pruning, logical encoding, transpose constraints and
kernel/packing amortization. This is an original scalar model, not PyTorch,
CUTLASS, cuSPARSELt or hipSPARSELt execution.

```powershell
python -m experiments.semi_structured_sparsity_selection.run
python -m experiments.semi_structured_sparsity_selection.run --check
python -m unittest tests.test_semi_structured_sparsity_selection
```

The fixture explicitly prunes adjacent row quartets, stores two values and
their ascending two-bit indices in a logical nibble, then compares a
packed-value product with the **same pruned dense** matrix. Its nibble
layout is not a backend physical format. Compression is not an implicit
pruning step. Zero values may occupy retained slots.

The row mask need not survive transpose. An independent greedy 4x4
bidirectional policy illustrates the changed pruning tradeoff; it is not
a reproduction or quality evaluation of upstream tile pruning.

The separate FP16/BF16 budget counts logical metadata and independently
padded forward/transpose orientations. Sparse-weight shape gates follow
the inspected backend configuration. Caller-side padding/cropping and
dense-operand columns aligned to eight are explicit model assumptions.
No actual backend packed bytes, peak HBM or model accuracy is reported.

Kernel times use hypothetical **dense-equivalent effective** TFLOP/s:
do not multiply by another marketing sparsity factor. One-time packing
must amortize over reuse; at one token the chosen padding/rate model has
no strict break-even. Rates/times are not measurements.

Sources:
[pinned sparse tensor classes](https://github.com/pytorch/pytorch/blob/5a7eae40861f6b8cc9ec71d3bc0e61f17a6568fa/torch/sparse/semi_structured.py)
and [conversion code](https://github.com/pytorch/pytorch/blob/5a7eae40861f6b8cc9ec71d3bc0e61f17a6568fa/torch/sparse/_semi_structured_conversions.py).
CUTLASS FP32 follows a different 1:2 branch; FP8 uses scaled operations
and additional platform gates. Neither is modeled. The August 2026 ROCm
FNUZ change is context, not a local accelerator result.

Upstream [LICENSE](https://github.com/pytorch/pytorch/blob/5a7eae40861f6b8cc9ec71d3bc0e61f17a6568fa/LICENSE)
rights retained; no implementation or weights copied. GitHub license
metadata returned `NOASSERTION`, not a confirmed single SPDX label.

**Synthetic; independently_reproduced=false.**
