# Round 123: kernel Cholesky, effective matrix identity and variance

Original scalar factor/triangular solves with independent Fraction
Gaussian elimination. No native linear_operator or GPU execution.
Fixed toy's diagonal shift is selected observation-noise variance,
not automatically numerical rescue jitter.

```powershell
python -m experiments.kernel_cholesky_contract_selection.run --check
python -m unittest discover -s tests -p test_kernel_cholesky_contract_selection.py -q
```

`--measure` deliberately replaces original CPU samples.
One unchanged factor versus refactor per each of eight RHS;
initial cached setup, kernel fitting/construction and native/GPU excluded.
Original full-symmetry/positive-pivot guard is explicit and narrower
than reproducing native trusted-triangle/rescue/tracing behavior.

linear_operator pin `1c5e24bc774352461be818b9d347e93ddee7d7f7`;
March 4 modern API migration separately dated, not new algorithm.
MIT, upstream rights retained; no implementation copied.

Figure-tier concept: existing original 60-solid magazine sorter
keeps matrix/factor identity and latent/noisy drawers distinct.
Static analogy; browser inspection not performed.

- [中文 FAQ](../../docs/components/kernel-cholesky-jitter-factor-reuse-variance-selection-faq.md)
- [English FAQ](../../docs/components/kernel-cholesky-jitter-factor-reuse-variance-selection-faq.en.md)
- [H5](../../docs/reports/kernel-cholesky-jitter-factor-reuse-variance-sizer.html)
- [Seven cards](../../docs/reports/kernel-cholesky-jitter-factor-reuse-variance-selection-cards.html)
- [Report](report.json) and [CPU samples](latency.json)
