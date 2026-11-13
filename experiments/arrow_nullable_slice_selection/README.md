# Round 108: nullable list offsets, validity and retained buffers

Original immutable little-endian List<int32> model; synthetic inputs and
separately measured Python CPU operations. `independently_reproduced=false`.
No native Arrow, native constructor failure, RSS, allocator or peak HBM.

- [中文 FAQ](../../docs/components/arrow-nullable-list-slice-buffer-retention-selection-faq.md)
- [English FAQ](../../docs/components/arrow-nullable-list-slice-buffer-retention-selection-faq.en.md)
- [H5 byte budget](../../docs/reports/arrow-nullable-list-slice-buffer-retention-sizer.html)
- [Seven cards](../../docs/reports/arrow-nullable-list-slice-buffer-retention-selection-cards.html)
- [Original SVG](../../assets/svg/arrow-nullable-list-slice-buffer-retention-selection-faq-summary.svg)

```powershell
python -m experiments.arrow_nullable_slice_selection.run
python -m unittest discover -s tests -p test_arrow_nullable_slice_selection.py -q
python -m experiments.arrow_nullable_slice_selection.run --check
```

`--measure` explicitly replaces sealed timings; ordinary generation and checks
never remeasure. Two warmups, seven alternating repeats, 25 operations per
repeat; each sample is mean milliseconds/operation, then medians across repeats.
Primary creates a view handle without reading rows. Baseline reads logical rows
and constructs validated canonical buffers: intentionally different work, not
a native Arrow speedup. Logical equality is checked outside timed loops.

## Evidence and bounds

Apply parent offset to row offsets and parent validity bits once.
Child positions are already absolute. Keep parent-null, valid-empty and
child-null distinct; never read null payload sentinel as logical data.
Shared views retain the root. Canonical copy explicitly discards hidden
null-parent spans and rebases offsets; it is our policy, not a pyarrow compact API.

H5 accepts explicit backing/visible counts and buffer-presence flags, computes
integer bitmap ceilings and rejects inexact/invalid inputs. Four/eight-byte
offset and one/two/four/eight-byte value budgets are arithmetic choices; the
executable data model only implements List<int32>. Selected payloads exclude
Python objects, intermediates, aliases, allocator behavior, RSS and HBM.

Pinned Arrow `0974e94ef5c2d510250923a7af239ec14a522700` is the November 13,
2026 required-object None guard change. Valid nullable buffers remain supported.
Apache-2.0 upstream rights retained; no implementation, dataset or assets copied.
Source references and narrower type/NumPy/flatten caveats are in the FAQs.
