# Round125: rational rate, centered FIR and anti-alias contracts

Original exact nonzero-phase FIR model and independent dense zero-
insertion convolution. Synthetic, independently_reproduced=false.
No native SciPy/default Kaiser/GPU/audio-quality reproduction.

```powershell
python -m experiments.polyphase_resampling_contract_selection.run --check
python -m unittest discover -s tests -p test_polyphase_resampling_contract_selection.py -q
```

`--measure` deliberately replaces separately measured original CPU
float samples. Fixed U3/D2, five taps, equal outputs checked; native
filter design/streaming/quality excluded.

Selected zero extension, odd symmetric exact FIRs and nonempty 1-D
input only. Identity reduced rate returns copy without filtering.
Toy fixed FIR is not suitable for arbitrary rates by default.

SciPy pin `30149047c624e328459a4d66d1e5f8b047b7bde7`,
tree2026-11-13; signaltools October23 change affects empty lfilter/
sosfilt inputs, not new resampling theory. BSD-3-Clause.
Original implementation/oracle/art; no upstream code copied.

Figure-tier sample-ticket sorter: base/four feet, three trays,
six rails,21 cards/21 tabs/four fasteners (60 solids).
Conceptual phases/selection/boundaries, not physical DSP hardware.
No browser visual inspection.

- [中文](../../docs/components/polyphase-rational-resampling-phase-antialias-selection-faq.md)
- [English](../../docs/components/polyphase-rational-resampling-phase-antialias-selection-faq.en.md)
- [H5](../../docs/reports/polyphase-rational-resampling-phase-antialias-sizer.html)
- [7 cards](../../docs/reports/polyphase-rational-resampling-phase-antialias-selection-cards.html)
- [Report](report.json), [CPU samples](latency.json)
