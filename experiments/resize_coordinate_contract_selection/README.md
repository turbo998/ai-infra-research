# Round126: integer extent, resize coordinates and nearest ties

Original Fraction1-Dgeometry/two-neighbor samples with independent
triangular hat-basis oracle. Synthetic, independently_reproduced=false.
No native ONNX/GPU/vision-quality execution.

```powershell
python -m experiments.resize_coordinate_contract_selection.run --check
python -m unittest discover -s tests -p test_resize_coordinate_contract_selection.py -q
```

`--measure` deliberately replaces original CPU float timings.
Cached half-pixel coordinates versus grid recomputation per eight
1-Dchannels. Cached setup,2-Dpipeline,native/GPU,antialias excluded.

Supports selected nonempty positive exact scale or integer size,
half_pixel/align_corners/asymmetric/pytorch_half_pixel,
clamped linear values and two nearest tie policies.
No cubic/ROI/exclude_outside/quantized dtype or native shape inference.

Pinned ONNX`3c41120b9c69657a1dd13407b080bd451ed06c9c`,
Sep30fix`a95e28ea0cf151cedefba7f94f2f22e7bf7febec`,
Apache-2.0; upstream rights retained,no implementation copied.

Figure-tier ticket grid archive: original60solids—base,four feet,
three trays,six rails,21cards,21tabs,four fasteners.
Static sampling-policy metaphor; browser inspection not performed.

- [中文](../../docs/components/resize-integer-size-half-pixel-align-corners-selection-faq.md)
- [English](../../docs/components/resize-integer-size-half-pixel-align-corners-selection-faq.en.md)
- [H5](../../docs/reports/resize-integer-size-half-pixel-align-corners-sizer.html)
- [7 cards](../../docs/reports/resize-integer-size-half-pixel-align-corners-selection-cards.html)
- [Report](report.json),[CPU records](latency.json)
