# Round 120: SparseAdam coalescing, mask and time

Original scalar table model; no native optimizer/tensor/GPU reproduction.
Independent Decimal update reference and Fraction grouping oracle.
Teaching hyperparameters lr=.1, betas=.5, eps=0 are not source defaults.
Positive finite denominator is an explicit narrower original policy.

```powershell
python -m experiments.sparse_adam_contract_selection.run --check
python -m unittest discover -s tests -p test_sparse_adam_contract_selection.py -q
```

`--measure` deliberately replaces original CPU coalescing samples;
excludes all moment/parameter updates and native sparse tensor operations.
H5 uses bounded integer gradient inputs for exact grouping sums;
its first-update toy is separate from row-sparse memory accounting.

Pinned PyTorch tree `22e0dfd2aed75aa2bc0a44e7ff4b49753f62c3ce`;
existing SparseAdam rules, not a November algorithm launch.
August 28 public functional API change separately dated.
BSD-style source license; no upstream implementation copied.

Figure-tier concept: existing original 60-solid magazine sorter
coalesces duplicate tickets but keeps full moment drawers.
Static analogy; browser visual inspection not performed.

- [中文 FAQ](../../docs/components/sparse-adam-coalescing-mask-step-state-selection-faq.md)
- [English FAQ](../../docs/components/sparse-adam-coalescing-mask-step-state-selection-faq.en.md)
- [H5](../../docs/reports/sparse-adam-coalescing-mask-step-state-sizer.html)
- [Seven cards](../../docs/reports/sparse-adam-coalescing-mask-step-state-selection-cards.html)
- [Report](report.json) and [CPU samples](latency.json)
