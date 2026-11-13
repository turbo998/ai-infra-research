# Round 106: sparse coordinate support, cached inverse maps and row identity

Original batched 1D scalar cross-correlation with no bias/nonlinearities.
Synthetic inputs, `independently_reproduced=false`; original CPU timing is
separately measured. No native spconv/CUDA, multi-channel 3D, model accuracy,
GPU performance or peak-HBM reproduction.

- [中文 FAQ](../../docs/components/sparse-convolution-coordinate-map-submanifold-inverse-selection-faq.md)
- [English FAQ](../../docs/components/sparse-convolution-coordinate-map-submanifold-inverse-selection-faq.en.md)
- [H5 budget](../../docs/reports/sparse-convolution-coordinate-map-submanifold-inverse-sizer.html)
- [Seven cards](../../docs/reports/sparse-convolution-coordinate-map-submanifold-inverse-selection-cards.html)
- [Original SVG](../../assets/svg/sparse-convolution-coordinate-map-submanifold-inverse-selection-faq-summary.svg)

```powershell
python -m experiments.sparse_conv_coordinate_map_selection.run
python -m unittest tests.test_sparse_conv_coordinate_map_selection -q
python -m experiments.sparse_conv_coordinate_map_selection.run --check
```

`--measure` explicitly replaces sealed CPU samples. Normal generation/checking
does not remeasure. Primary reuses an immutable submanifold map and excludes
initial construction; baseline builds the same map. Both check exact ordered
coordinates and apply the same scalar weights. CPU ratios are not native GPU
speedup or end-to-end inference claims.

## Scope and evidence

Regular and submanifold paths match independent dense scans at their respective
declared output sites, but their second-layer values differ. Zero features do
not deactivate coordinates. Batch identity and row ordering are part of the map.
Duplicate coordinates are rejected rather than silently coalesced.

Reversing saved pairs with tied weights gives a checked linear adjoint on original
support, not a numerical inverse or the support of ordinary transpose.
Separately learned inverse weights need not be adjoint. Our exact ordered-row
guard is stronger than inspected source shape/count checks; no native bug is claimed.
The toy accepts an empty map, without reproducing native vanished-point errors.

The H5 uses synthetic input/output counts and valid pairs on a shared declared
grid volume. Native `[2,K,N]` is one selected pair-array path, not implicit-GEMM
allocation. Compact pair IDs omit grouping metadata; workspace, masks,
sorting, allocator and peak HBM are excluded.

Pinned spconv `263d6b47425ef843c82f997b12d8b714013d216c` is from December 15,
2024; inspected `conv.py` last changed March 23, 2023. These longstanding
mechanics are not a 2026 launch. Apache-2.0 rights remain with upstream holders;
no source implementation, weights or datasets are copied.
