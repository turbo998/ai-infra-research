# Round 104: Gaussian tile compositing and wide intersection offsets

Original same-finite-support dense/tile CPU compositing, independent
product-form weights and effective-alpha gradients, exclusive cutoff
controls, accumulated small alphas and synthetic wide-offset arithmetic.
No 3D projection, source ellipse culling, gsplat/CUDA or image-quality
benchmark is reproduced.

```powershell
python -m experiments.gaussian_tile_rasterization_selection.run
python -m experiments.gaussian_tile_rasterization_selection.run --check
python -m unittest tests.test_gaussian_tile_rasterization_selection
```

`latency.json` seals two warmups and seven alternating CPU repeats.
Rendering/checking never remeasures; explicit `--measure` replaces
samples and requires updating timing prose. `--check --measure` is invalid.
Both timed paths disable early termination and include validation.

See [English FAQ](../../docs/components/gaussian-splat-tile-alpha-wide-offset-selection-faq.en.md),
[中文 FAQ](../../docs/components/gaussian-splat-tile-alpha-wide-offset-selection-faq.md),
[H5](../../docs/reports/gaussian-splat-tile-alpha-wide-offset-sizer.html),
[seven cards](../../docs/reports/gaussian-splat-tile-alpha-wide-offset-selection-cards.html)
and [SVG](../../assets/svg/gaussian-splat-tile-alpha-wide-offset-selection-faq-summary.svg).

Pinned gsplat sources retain Apache-2.0 rights; no implementation or
scene assets were copied. CPU float64 is not bitwise source float32
parity near thresholds. Logical sort buffers exclude actual workspace
and peak allocation; 64-bit offsets do not widen every remaining index.
