# Round 117: categorical filter law and request identity

Original rational filters/CDF and supplied-uniform identity model.
No native FlashInfer, Philox or canonical float-tree reproduction.

```powershell
python -m experiments.sampling_filter_identity_selection.run --check
python -m unittest discover -s tests -p test_sampling_filter_identity_selection.py -q
```

`--measure` deliberately replaces original CPU timing. Routine checks
reuse sealed samples. Timed work is precomputed integer-CDF lookup:
C-backed stdlib bisect versus Python linear loops, 32 supplied draws.
Filtering, sorting, CDF building, random generation and GPU excluded.

Original boundaries: exact rational nonnegative weights, positive mass,
1<=k<=V, 0<p<=1, 0<=u<1. Stable exact-k token-ID tie policy is explicit,
not native tie compatibility. H5 uses BigInt for exact ratio comparison.
The fast-path bound control is semantic only and does not execute the
source's >=65536-vocabulary optimized path.

FlashInfer pin `9b69c0e18a13c353bd6229d70780fd14dd19ff82`,
November 13, 2026, Apache-2.0. New basic batch-invariant categorical
sampling is distinguished from existing filtered APIs. No upstream
implementation copied.

Figure-tier concept: the existing original 60-solid magazine sorter
routes distribution and request tickets. Static analogy only;
browser visual inspection not performed.

- [中文 FAQ](../../docs/components/categorical-topk-topp-request-identity-selection-faq.md)
- [English FAQ](../../docs/components/categorical-topk-topp-request-identity-selection-faq.en.md)
- [Seven cards](../../docs/reports/categorical-topk-topp-request-identity-selection-cards.html)
- [H5](../../docs/reports/categorical-topk-topp-request-identity-sizer.html)
- [Report](report.json) and [CPU records](latency.json)
