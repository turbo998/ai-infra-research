# Round124: regional capture ownership and layout

Original bounded scalar storage/view/event model; no PyTorch execution.
Synthetic, independently_reproduced=false. Separately measured
original CPU records are not compiler/native/GPU benchmarks.

```powershell
python -m experiments.regional_capture_snapshot_selection.run --check
python -m unittest discover -s tests -p test_regional_capture_snapshot_selection.py -q
```

`--measure` deliberately replaces sealed CPU samples.
Both paths include finite backing validation; detach and clone have
different ownership semantics. Eight tests cover independent index
oracles, identity/alias guards, module calls and JS parity.

Supports only nonempty nonnegative-stride 1-D finite scalar views,
acyclic list/tuple/string-keyed dictionaries and primitives.
Native arbitrary objects, sparse tensors, autograd, dtype/subclasses,
hooks/export/compilation are not reproduced.

PyTorch source pin `7bbc31dd45f4e43a5e3899e7e328aedfd967f2ff`,
2026-11-14T00:44:56Z, BSD-style license. Existing upstream rights
retained; no implementation copied.

Figure-tier concept: original 60-solid magazine/ticket archive maps
live references versus frozen copies; a static analogy, not tensor
hardware. Parts: four feet, base, three trays, six rails,21 cards,
21 tabs and four fasteners. Browser visual inspection not performed.

- [中文](../../docs/components/regional-aoti-capture-detach-clone-storage-contract-selection-faq.md)
- [English](../../docs/components/regional-aoti-capture-detach-clone-storage-contract-selection-faq.en.md)
- [H5](../../docs/reports/regional-aoti-capture-detach-clone-storage-contract-sizer.html)
- [Seven cards](../../docs/reports/regional-aoti-capture-detach-clone-storage-contract-selection-cards.html)
- [Report](report.json), [CPU samples](latency.json)
