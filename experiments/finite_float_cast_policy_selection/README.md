# Finite FP4/FP6 RNE,signed zero and NaN policy

Original exact-rational saturating conversion with finite codebook decoding,even-code midpoint ties,signed zero and explicit NaN block.No native ONNX casts.No float32 input pre-rounding,block scales,calibration or quality metrics.

```powershell
python -m experiments.finite_float_cast_policy_selection.run --measure
python -m experiments.finite_float_cast_policy_selection.run --check
python -m unittest discover -s tests -p test_finite_float_cast_policy_selection.py -q
```

Source change is a Nov13BJTdocumentation/test clarification:NaNcast result unspecified,not a new native algorithm.Original reject-NaN policy is not a backend exception guarantee.Inf/NaNerrors stay null and are excluded from finite-error denominators.All raw finite patterns/midpoints have original mathematical coverage.Native NaN/GPU/runtime peak/model quality remain unknown.

[English FAQ](../../docs/components/onnx-finite-fp4-fp6-cast-rne-nan-policy-selection-faq.en.md) / [中文](../../docs/components/onnx-finite-fp4-fp6-cast-rne-nan-policy-selection-faq.md).Original60-solid figure;visual browser inspection not performed.
