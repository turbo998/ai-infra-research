# Round 105: multiresolution hash-grid collision adjoints and LoD

Original standard-library CPU references, synthetic scalar fields and separately
measured CPU timings. `independently_reproduced=false`; no native tiny-cuda-nn,
GPU performance, neural-field training or reconstruction-quality claim.

- [中文 FAQ](../../docs/components/multires-hash-grid-collision-lod-selection-faq.md)
- [English FAQ](../../docs/components/multires-hash-grid-collision-lod-selection-faq.en.md)
- [Interactive H5](../../docs/reports/multires-hash-grid-collision-lod-sizer.html)
- [Seven cards](../../docs/reports/multires-hash-grid-collision-lod-selection-cards.html)
- [Original SVG](../../assets/svg/multires-hash-grid-collision-lod-selection-faq-summary.svg)

```powershell
python -m experiments.multires_hashgrid_lod_selection.run
python -m unittest tests.test_multires_hashgrid_lod_selection -q
python -m experiments.multires_hashgrid_lod_selection.run --check
```

`--measure` explicitly replaces `latency.json`; ordinary generation/checking
uses sealed samples. Timings include validation. Direct gathers build Python
corner tuples containing derivative coefficients; the unfolded baseline
constructs its 17×17 scalar corner field once per batch. These untuned CPU
implementation costs are not a native/GPU algorithm comparison.

## Evidence and limits

Four bilinear corners share two table buckets; VJP accumulates every occurrence.
Independent unfolded tied-field interpolation and table/coordinate finite
differences check the original references. Collision overwrite and missing/double
soft weights are negative controls. Hard/soft epsilon boundaries use mathematical
float64, not bitwise source float32 behavior.

The fixed-LoD raw-feature squared-loss Hessian is checked, but source LoD-input
derivatives and complete double backward are not implemented. The bounded
calculator uses integer growth one/two, 2D/3D and resolution <=65536.
It models aligned/capped hash entries and unpadded output/base-output buffers;
ideal dense vertices are not a supported native allocation.
Gradients, optimizer states, layout staging, allocator and peak HBM are excluded.

The September 29, 2026 native multilevel LoD capability is pinned at
`3daa6e5b8b8e0fa17971a1a20acfbb3754494b38`. Actual `LICENSE.txt` has
BSD-3-Clause terms; GitHub's API reports `NOASSERTION`. Upstream rights remain
with their holders; no source implementation or weights are copied.
