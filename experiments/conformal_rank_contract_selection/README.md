# Round 122: split-conformal finite ranks and unbounded boundaries

Original exact-rank/exhaustive-holdout model, not native MAPIE.
Fixed synthetic score world and frozen score mechanism; no real-data
or per-group coverage claim. Explicit tagged unbounded result, never zero.

```powershell
python -m experiments.conformal_rank_contract_selection.run --check
python -m unittest discover -s tests -p test_conformal_rank_contract_selection.py -q
```

`--measure` deliberately replaces original CPU selection samples.
Precomputed integer scores, alpha=1/5; excludes model/score fitting,
generation, inference and native/GPU work.
Native APIs can reject insufficient samples or explicitly allow
unbounded outputs; different wrappers are not all reproduced.

MAPIE pin `d5ea5e07aea09a897726fd8641c193850a5c540d`,
November 10, 2026 educational update; existing conformal theory.
BSD-3-Clause, upstream rights retained; no implementation copied.

Figure-tier concept: existing original 60-solid magazine sorter
reserves the extra test rank and an explicit overflow tray.
Static analogy; no browser visual inspection.

- [中文 FAQ](../../docs/components/split-conformal-finite-rank-unbounded-boundary-selection-faq.md)
- [English FAQ](../../docs/components/split-conformal-finite-rank-unbounded-boundary-selection-faq.en.md)
- [H5](../../docs/reports/split-conformal-finite-rank-unbounded-boundary-sizer.html)
- [Seven cards](../../docs/reports/split-conformal-finite-rank-unbounded-boundary-selection-cards.html)
- [Report](report.json) and [CPU samples](latency.json)
