# Round 110: count-weighted synchronized moments and gradients

Original one-channel scalar moments/gradient model with exact rational and
finite-difference references. Synthetic, `independently_reproduced=false`;
original Python CPU measurements are not native PyTorch or network timings.

- [中文 FAQ](../../docs/components/sync-batchnorm-count-weighted-moments-selection-faq.md)
- [English FAQ](../../docs/components/sync-batchnorm-count-weighted-moments-selection-faq.en.md)
- [H5 payload/representation calculator](../../docs/reports/sync-batchnorm-count-weighted-moments-sizer.html)
- [Seven cards](../../docs/reports/sync-batchnorm-count-weighted-moments-selection-cards.html)
- [SVG](../../assets/svg/sync-batchnorm-count-weighted-moments-selection-faq-summary.svg)

```powershell
python -m experiments.sync_batchnorm_moments_selection.run
python -m unittest discover -s tests -p test_sync_batchnorm_moments_selection.py -q
python -m experiments.sync_batchnorm_moments_selection.run --check
```

`--measure` explicitly replaces sealed samples. Otherwise no remeasurement.
Two warmups, seven alternating repeats, ten operations per sample; chart
uses median repeat means. Primary merges four precomputed summaries;
baseline rescans features twice. Initial summaries, movement, network and
native kernels are excluded. No end-to-end speedup claim.

## Semantics and boundaries

Count weighting alone does not repair local variance: retain the between-mean
M2 correction. Empty summaries are numerical identities, but empty ranks
still participate in the inspected ordinary forward/backward protocol.
All-empty global normalization rejects; undefined estimators are None.
The native CUDA Graph filtering branch differs and is not certified.

Global forward requires global backward sums for input derivatives. Scalar
loss is SUM, with explicit epsilon/gamma/beta; DDP parameter-gradient scaling
is not implemented. Population and unbiased variances use separate
denominators. Original model carries Python counts and M2, not the native
finite-precision invstd packet.

H5 counts observations per channel, including spatial dimensions. Selected
float count exactness and saved int32 fit are separate limited checks.
Packet, gather-buffer and backward payloads are not wire bytes, native
collective time or peak HBM.

Pinned PyTorch `e5e96ce34777fb56e66232669a221c51955a57ef`, July 29, 2026;
wrapper collective API migration June 4. Longstanding mechanism, not a
November launch. Primary LICENSE terms inspected; GitHub SPDX NOASSERTION.
Upstream rights retained; no implementation, weights or datasets copied.
