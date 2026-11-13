# Round 119: actual backward fan-in and explicit precision

Original scalar task-slot and IEEE rounding model, not native autograd.
Different slots represent separate logical inputs, not actual tensor views.
`None` is undefined; zero is defined. A strict parent is inherited by OR.
Separate tasks do not guard external leaf `.grad` accumulation.

```powershell
python -m experiments.autograd_fanin_contract_selection.run --check
python -m unittest discover -s tests -p test_autograd_fanin_contract_selection.py -q
```

`--measure` deliberately replaces original CPU samples. Wide float64/final
float32 versus per-add struct float32 intentionally produce different results.
Includes input conversion; excludes native engine, tensor views and GPU.
No transactional rollback or general precision/memory guarantee.

Pinned PyTorch `22e0dfd2aed75aa2bc0a44e7ff4b49753f62c3ce`,
November 12, 2026. BSD-style upstream license; no implementation copied.
Source new flag distinguishes same-task fan-in from cross-call `.grad`.

Figure-tier concept: existing original 60-solid magazine sorter separates
gradient tickets before explicit combination. Four feet, base, three trays,
six rails, 21 cards, 21 tabs and four fasteners. Static analogy only;
browser inspection not performed.

- [中文 FAQ](../../docs/components/autograd-implicit-fanin-explicit-precision-selection-faq.md)
- [English FAQ](../../docs/components/autograd-implicit-fanin-explicit-precision-selection-faq.en.md)
- [H5](../../docs/reports/autograd-implicit-fanin-explicit-precision-sizer.html)
- [Seven cards](../../docs/reports/autograd-implicit-fanin-explicit-precision-selection-cards.html)
- [Report](report.json) and [CPU samples](latency.json)
