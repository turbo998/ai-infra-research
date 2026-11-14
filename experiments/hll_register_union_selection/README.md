# HLL register union and estimator boundary

Original finite-digest register model, independent binary-string oracle, classic harmonic/linear-counting teaching estimator, payload budget, Node parity and CPU rank-path samples. Pinned Redis rank/MAX contracts inform the discussion; native Murmur/Ertl, storage and accuracy are not reproduced.

```powershell
python -m experiments.hll_register_union_selection.run --measure
python -m experiments.hll_register_union_selection.run --check
python -m unittest discover -s tests -p test_hll_register_union_selection.py -q
```

The affine 16-bit identity permutation is deliberately not a uniform-hash accuracy claim: its teaching estimate 57.995698 differs substantially from the true union 30. MAX union correctness does not fix that statistical mismatch. Native fields remain null. See the [English FAQ](../../docs/components/hll-register-max-union-cardinality-estimator-selection-faq.en.md) and [中文 FAQ](../../docs/components/hll-register-max-union-cardinality-estimator-selection-faq.md).
