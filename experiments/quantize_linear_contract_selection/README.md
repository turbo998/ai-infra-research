# Round 121: integer QuantizeLinear operation and axis contracts

Original exact Fraction/BigInt per-axis int8/uint8 model.
Nearest-even precedes integer zero point, then saturation.
Rank-two only; bounded int32 quotient; no native float precision,
runtime overflow, scalar/blocked or FP6/FP8/INT4 reproduction.

```powershell
python -m experiments.quantize_linear_contract_selection.run --check
python -m unittest discover -s tests -p test_quantize_linear_contract_selection.py -q
```

`--measure` deliberately replaces original CPU Fraction samples.
Reciprocal construction included; exact outputs match division.
No floating reciprocal equivalence or consistent speedup inferred.

ONNX tree `3c41120b9c69657a1dd13407b080bd451ed06c9c`
November 12, 2026 UTC. Reference's August 29 float6 update is
separately dated and not evaluated. Apache-2.0, upstream rights retained;
no upstream implementation copied.

Figure-tier concept: existing original 60-solid magazine sorter
routes ratio tickets through ordered rounding/offset/saturation stages.
Static analogy; browser visual inspection not performed.

- [中文 FAQ](../../docs/components/quantize-linear-rne-zero-point-axis-selection-faq.md)
- [English FAQ](../../docs/components/quantize-linear-rne-zero-point-axis-selection-faq.en.md)
- [H5](../../docs/reports/quantize-linear-rne-zero-point-axis-sizer.html)
- [Seven cards](../../docs/reports/quantize-linear-rne-zero-point-axis-selection-cards.html)
- [Report](report.json) and [CPU samples](latency.json)
