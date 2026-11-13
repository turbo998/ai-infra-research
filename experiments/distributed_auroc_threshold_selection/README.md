# Round 111: distributed AUROC, ties and threshold-state approximation

Original binary grouped ROC with rational coordinates and an independent
pairwise oracle. Synthetic, `independently_reproduced=false`; original Python
CPU timing only, not TorchMetrics, distributed execution or real model quality.

- [中文 FAQ](../../docs/components/distributed-auroc-ties-threshold-state-selection-faq.md)
- [English FAQ](../../docs/components/distributed-auroc-ties-threshold-state-selection-faq.en.md)
- [H5 state/workspace budget](../../docs/reports/distributed-auroc-ties-threshold-state-sizer.html)
- [Seven cards](../../docs/reports/distributed-auroc-ties-threshold-state-selection-cards.html)
- [SVG](../../assets/svg/distributed-auroc-ties-threshold-state-selection-faq-summary.svg)

```powershell
python -m experiments.distributed_auroc_threshold_selection.run
python -m unittest discover -s tests -p test_distributed_auroc_threshold_selection.py -q
python -m experiments.distributed_auroc_threshold_selection.run --check
```

`--measure` explicitly replaces sealed samples. Otherwise no remeasurement.
Two warmups, seven alternating repeats, three operations/sample; table is
median repeat means. Both algorithms validate input. Fraction arithmetic
loses to scalar pairwise work at the two smallest tiers; raw timings retained.
No native optimization or distributed speedup claim.

## Contracts and boundaries

Do not average completed local AUROCs: cross-rank positive-negative pairs
matter. Exact tied groups match pairwise half credit. Fixed grids merge
TN/FP/FN/TP by integer SUM with score >= threshold. Identical-grid state
merging is exact, but grid approximation is not exact global ranking.

The inspected binned rule does not insert an origin. Score-one endpoint
controls and the separately labeled original origin-completion variant are
not native failure/fix reproductions. Original missing-class handling raises
explicitly and differs from the native warning/zero-rate path.

Persistent int64 T*2*2 state excludes update workspace. A selected N*T int64
matrix is counted only in the inspected <=50k vectorized branch; larger
threshold-loop matrix size is left null, not zero. Raw payloads exclude
internal variable-size padding, sorting, transport and allocator. No peak HBM.

Pinned TorchMetrics `e843b84bb9e957514deb8971340ec6c6e3f0887c`, September 8,
2026 compatibility cleanup; ROC mechanics are longstanding. Apache-2.0
upstream rights retained. No library implementation, weights or datasets copied.
