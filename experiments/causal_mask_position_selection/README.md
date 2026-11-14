# Round130: non-square causal alignment,positions and boolean polarity

Original exact position/triangular masks and uniform-zero-logit scalar
means,independent sorted-position oracle and separately measured original
CPU paths. Synthetic,independently_reproduced=false.
No native SDPA/GPU,softmax,dropout,model-quality or cache-kernel reproduction.

```powershell
python -m experiments.causal_mask_position_selection.run --check
python -m unittest discover -s tests -p test_causal_mask_position_selection.py -q
```

`--measure` deliberately replaces original CPU timing records.
Eight queries,prefix sums rebuilt every operation versus repeated dense
coordinate filtering. Two warmups,seven alternating repeats,three
operations/sample;not general attention or GPU performance.

PyTorchpin`7bbc31dd45f4e43a5e3899e7e328aedfd967f2ff`,BSD-style.
SDPAupper-leftversuslower-rightsuffixalignment;
SDPATrueallowversusMHATrueblocked.
PinnedNov14treeupdateisregionalAOTI,notnewmasktheory.
No upstream implementation copied.
Empty rows explicitly tagged null in original model;native behavior unknown.

StaticFigure-tieroriginal60-solidticketcabinet:
base,fourfeet,threetrays,sixrails,21cards,21tabs,fourfasteners.
Reusedoriginalgeometryasposition-ticketmetaphor;
no browser inspection performed after prior approval refusal.

- [中文](../../docs/components/non-square-causal-mask-cache-position-polarity-selection-faq.md)
- [English](../../docs/components/non-square-causal-mask-cache-position-polarity-selection-faq.en.md)
- [H5](../../docs/reports/non-square-causal-mask-cache-position-polarity-sizer.html)
- [7 cards](../../docs/reports/non-square-causal-mask-cache-position-polarity-selection-cards.html)
- [Report](report.json),[CPU records](latency.json)
