# Dense ConvTranspose finite-domain output padding

Original onechannel integer tap scatter, independent dense forward-matrix oracle and finite-domain adjoint checks. Output padding selects a potentially nonzero boundary sample rather than appending zeros; the transpose is not a numerical inverse.

```powershell
python -m experiments.transpose_output_padding_selection.run --measure
python -m experiments.transpose_output_padding_selection.run --check
python -m unittest discover -s tests -p test_transpose_output_padding_selection.py -q
```

The model deliberately restricts output_padding below stride for matching finite forward-matrix dimensions, narrower than all native dilation-related configurations. No bias/grouped execution; grouped byte accounting is a separate logical budget. Python timings against an educational dense matrix do not measure native PyTorch/BLAS/cuDNN/GPU or quality. [English FAQ](../../docs/components/dense-transpose-output-padding-finite-adjoint-selection-faq.en.md) / [中文 FAQ](../../docs/components/dense-transpose-output-padding-finite-adjoint-selection-faq.md).
