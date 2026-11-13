# Round 118: stable float radix-key contracts

Original float32 comparison-key and stable byte-pass model.
Retains original payload bits/IDs; explicitly rejects NaNs.
No native CUB/GPU execution or internal zero-encoding compatibility.

```powershell
python -m experiments.float_radix_contract_selection.run --check
python -m unittest discover -s tests -p test_float_radix_contract_selection.py -q
```

`--measure` deliberately replaces CPU samples. Routine checks reuse
sealed data. Original Python byte buckets versus C-backed Python sorted,
precomputed uint32 keys; includes implementation/validation differences.
Float conversion, input generation, transfer and native execution excluded.

Pinned NVIDIA/cccl `0dd2af496098ca890304ed4b8b3bc13e6b88851c`,
November 10, 2026. CUB header SPDX BSD-3; upstream rights retained.
Recent ADL-maintenance change is not a new floating-order algorithm.
No upstream implementation copied.

Figure-tier concept: original 60-solid magazine sorter routes
comparison tickets while retaining original payload identity.
Four feet, one base, three trays, six rails, 21 cards, 21 tabs,
four fasteners. Static analogy; no browser inspection.

- [中文 FAQ](../../docs/components/float-radix-signed-zero-transformed-bit-range-selection-faq.md)
- [English FAQ](../../docs/components/float-radix-signed-zero-transformed-bit-range-selection-faq.en.md)
- [H5](../../docs/reports/float-radix-signed-zero-transformed-bit-range-sizer.html)
- [Seven cards](../../docs/reports/float-radix-signed-zero-transformed-bit-range-selection-cards.html)
- [Report](report.json) and [CPU samples](latency.json)
