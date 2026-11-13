# Round 115: DLPack description, managed-record lifetime and stream order

Original Python models, not native DLPack/PyCapsule/C ABI execution.

```powershell
python -m experiments.dlpack_lifetime_stream_selection.run --check
python -m unittest discover -s tests -p test_dlpack_lifetime_stream_selection.py -q
```

`--measure` deliberately replaces original CPU timing. Normal checks
reuse sealed samples. Timed work is readonly Python memoryview creation
versus immutable bytes copying, not native interop/GPU synchronization.
Previous-path output cleanup is outside the next path's timing;
within-path replacement is included. Owner creation and checks excluded.

Eight tests cover independent address walks, scalar/empty/legacy/reverse
layouts, one-shot capsules, explicit release orders, major mismatch,
readonly alias versus snapshot, full-permutation schedule oracle and
Node/Python budget parity. Address enumeration is bounded to 100,000
entries; H5 accounting does not allocate selected tensors.

DLPack pin `c1954b9becbd37d6c04e705cfb13e09331a9d762`,
November 4, 2026, Apache-2.0: version macro 1.3→1.4 only.
Existing lifetime/stream rules are not new. The header's >=1.2 non-null
stride rule is distinguished from older null-stride Python prose.
No upstream implementation copied.

Figure-tier concept: the repository's original 60-solid magazine
sorter is a borrow/return ticket cabinet. Static analogy only;
browser visual inspection not performed.

- [中文 FAQ](../../docs/components/dlpack-versioned-capsule-strides-stream-lifetime-selection-faq.md)
- [English FAQ](../../docs/components/dlpack-versioned-capsule-strides-stream-lifetime-selection-faq.en.md)
- [Seven cards](../../docs/reports/dlpack-versioned-capsule-strides-stream-lifetime-selection-cards.html)
- [H5](../../docs/reports/dlpack-versioned-capsule-strides-stream-lifetime-sizer.html)
- [Report](report.json) and [CPU records](latency.json)
