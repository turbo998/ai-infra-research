# Round158: rounded SiLU bit-pattern lookup/cache selection

Original finite FP32/16-bit arithmetic and independent declared cache ledger;not CUDA/libdevice/full-domain byte reproduction.

```powershell
python -m experiments.rounded_silu_bit_table_selection.run --check
python -m unittest discover -s tests -p test_rounded_silu_bit_table_selection.py -q
```

`--measure` replaces local CPU samples;ordinary generation retains sealed samples.The native table has65536entries/128KiB;this original dictionary has8finite keys/dtype.Table construction and every GPUcost are excluded from CPUtiming.Unknown native evidence remains null.

Single-final-round misses7/16finite lanes;bit-key lookup misses0/16.Signed numeric zero keys lose1/2word identities.BF16−89 distinguishes FP32exp overflow from stable float64arithmetic.Warm exact dtype/device cache precedes capture probes;empty uncaptured warmup can still allocate a logical table.

[中文 FAQ](../../docs/components/rounded-silu-bit-pattern-lut-capture-cache-selection-faq.md) / [English FAQ](../../docs/components/rounded-silu-bit-pattern-lut-capture-cache-selection-faq.en.md).Inf/NaN input words are explicitly outside the finite prototype,not outside native scope.No browser visual acceptance.
