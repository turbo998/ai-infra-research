# Round 116: mel-filter rounding, axes, scales and frame contracts

Original scalar mel-filter/projection models. No native audio backend,
librosa parity reproduction, decoding, resampling, STFT, ASR or GPU run.

```powershell
python -m experiments.mel_frontend_contract_selection.run --check
python -m unittest discover -s tests -p test_mel_frontend_contract_selection.py -q
```

`--measure` intentionally replaces original CPU projection timing;
routine checks preserve sealed samples. Timed nonzero-weight versus
dense scalar projections exclude bank construction/compression and
the entire native audio pipeline.

Eight focused tests include rational triangle and explicit frame-start
oracles, float32 cast order, axis rejection, frame permutation, dense
versus sparse equality, log units and Node/Python accounting parity.
Model-only all-zero bands raise explicitly; source helper warns.
The calculator selects even FFT grids and fixed float32 coefficients;
power/mel payload widths are selected assumptions, not native precision.

Transformers pin `880561aabb68be86c4c0f1c37ee5281e07ed1439`,
November 12, 2026, Apache-2.0. Selected model bridges explicitly preserve
Slaney normalization/scale, float32 cast order and contiguous transpose
while dropping librosa dependencies. Source-native bit-exactness is not
independently executed. No upstream implementation copied.

Figure-tier concept: the existing original 60-solid magazine sorter
becomes a frequency-to-mel weight cabinet. Static analogy only;
browser visual inspection not performed.

- [中文 FAQ](../../docs/components/mel-filterbank-cast-order-axis-frame-contract-selection-faq.md)
- [English FAQ](../../docs/components/mel-filterbank-cast-order-axis-frame-contract-selection-faq.en.md)
- [Seven cards](../../docs/reports/mel-filterbank-cast-order-axis-frame-contract-selection-cards.html)
- [H5](../../docs/reports/mel-filterbank-cast-order-axis-frame-contract-sizer.html)
- [Report](report.json) and [CPU records](latency.json)
