# Round 102: set-prediction matching and global object normalization

Original exhaustive assignment and independent query-scanning target-subset
DP, scalar fixed-assignment derivatives, permutation checks and DDP count
arithmetic. They do not implement Hungarian or reproduce detector
focal/GIoU losses, model quality or native runtime.

```powershell
python -m experiments.set_prediction_matching_selection.run
python -m experiments.set_prediction_matching_selection.run --check
python -m unittest tests.test_set_prediction_matching_selection
```

`latency.json` seals original CPU samples: two warmups, seven alternating
repeats. Rendering/checking never remeasures. Explicit `--measure`
replaces samples; update corresponding timing prose for any new run.
`--check --measure` is invalid.

See [English FAQ](../../docs/components/set-prediction-hungarian-matching-global-normalization-selection-faq.en.md),
[中文 FAQ](../../docs/components/set-prediction-hungarian-matching-global-normalization-selection-faq.md),
[H5](../../docs/reports/set-prediction-hungarian-matching-global-normalization-sizer.html),
[seven cards](../../docs/reports/set-prediction-hungarian-matching-global-normalization-selection-cards.html)
and [SVG](../../assets/svg/set-prediction-hungarian-matching-global-normalization-selection-faq-summary.svg).

Pinned RT-DETR sources retain their Apache-2.0 rights; no implementation
or model weights were copied. Array budgets are uniform-target logical
payloads, not measured allocations or peak memory.
