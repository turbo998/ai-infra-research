# Round 114: RoPE layout and partial-prefix contracts

Original scalar float64 plain-RoPE rotations and an independent complex
relative-angle oracle. No native FlashInfer/CUDA or checkpoint execution.

```powershell
python -m experiments.rope_layout_partial_dimension_selection.run --check
python -m unittest discover -s tests -p test_rope_layout_partial_dimension_selection.py -q
```

`--measure` deliberately replaces original CPU timing; routine checks do
not remeasure. Coefficients/matrices are precomputed for scalar paired
rotation versus dense matrix application. This is not GPU performance.

Eight-value heads rotate a six-value prefix. A shared Q/K permutation
makes adjacent and split-half layouts equivalent; flag-only switching
changes cross-position scores while both same-position checks pass.
Post-RoPE keys must not rotate twice. Tail identity, multiple positions,
dimension/coefficient guards and Node/Python bytes are covered.

The pinned cached API explicitly requires float32 coefficients; H5
enforces four-byte cache values. Selected Q/K payload is not additional
allocation, traffic, peak HBM or certified native dtype/shape support.
Signed positions in analytic tests are not valid cache-index guarantees.

Figure-tier concept: the repository's original 60-solid card sorter
aligns query/key coordinate magazines. Static analogy only; browser
visual inspection was not performed.

- [中文 FAQ](../../docs/components/rope-interleaved-neox-partial-rotary-layout-selection-faq.md)
- [English FAQ](../../docs/components/rope-interleaved-neox-partial-rotary-layout-selection-faq.en.md)
- [Seven cards](../../docs/reports/rope-interleaved-neox-partial-rotary-layout-selection-cards.html)
- [H5 budget](../../docs/reports/rope-interleaved-neox-partial-rotary-layout-sizer.html)
- [Report](report.json) and [CPU samples](latency.json)

FlashInfer pin `a984f0f6e8c950571de8fbb12f5e0fd948ad1b0b`
(2026-11-10), Apache-2.0. Related regular-API guard and cached-kernel
repair are distinguished; no native failure/fix reproduction claimed.
