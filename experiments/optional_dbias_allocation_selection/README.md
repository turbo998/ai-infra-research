# Unrequested dbias must not materialize a dense placeholder

Original exact fixed/varlen byte subledger for the Nov14ROCm CKbackward correction.No ROCm/PyTorch/CK/GPUallocation or gradient verification.An undefined tensor is not a dense zero gradient;provided caller memory stays live.

```powershell
python -m experiments.optional_dbias_allocation_selection.run --measure
python -m experiments.optional_dbias_allocation_selection.run --check
python -m unittest discover -s tests -p test_optional_dbias_allocation_selection.py -q
```

BigInt/decimal strings preserve counts beyond2^53.Selected6GiBother-live metadata and8GiBbudget are synthetic,not native peak/OOMprediction.Requested metadata validates presence only.CPUtoy caps at512KiBand zero-fills bytearray,unlike native at::empty.

[English FAQ](../../docs/components/rocm-ck-flash-backward-optional-dbias-allocation-selection-faq.en.md) / [中文](../../docs/components/rocm-ck-flash-backward-optional-dbias-allocation-selection-faq.md).Original60-solid figure;no visual browser inspection.
