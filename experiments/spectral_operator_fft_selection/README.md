# Real-grid spectral operator contracts / 实数网格频谱层契约

Round 98: an original standard-library two-dimensional DFT model validates
real-FFT storage, staged Hermitian boundary handling, explicit odd output
lengths and paired normalization against independent spatial circular
convolution and a four-loop full-complex inverse.

```powershell
python -m experiments.spectral_operator_fft_selection.run
python -m experiments.spectral_operator_fft_selection.run --check
python -m unittest tests.test_spectral_operator_fft_selection
```

The numerical fixture has one input/output channel and a fixed real circular
kernel. The kernel's unnormalized transform defines a fixed multiplier across
normalization conventions. Low-pass mode truncation is an intentional operator
change and is checked against the same retained spectrum, not called full
operator parity. Error remains null when an inferred inverse length changes
the output shape.

The separate H5 budget uses dense multi-channel complex weights and selected
input/output spectrum buffers. It is not peak HBM: FFT workspace, activations,
gradients, optimizer state and framework allocation are excluded. Its one-pass
byte-read formula is not GPU execution latency. PDE quality is unmeasured.

First-party context:
[pinned neuraloperator SpectralConv](https://github.com/neuraloperator/neuraloperator/blob/a6648eeeb60fe0837f2d7d021517a945a159a0c8/neuralop/layers/spectral_convolution.py),
including default `fft_norm="forward"`, final-axis mode conversion and
staged Hermitian enforcement. The
[February 2026 source change](https://github.com/neuraloperator/neuraloperator/commit/dcd46cd80b41d71e6feb258cfe4fdd2f524adce0)
documents a GPU concern; this harness does not reproduce that hardware
symptom or execute neuraloperator/PyTorch/cuFFT. Upstream MIT rights retained;
no source implementation copied.

**Synthetic; independently_reproduced=false.** No trained neural operator,
scientific-simulation error score or measured accelerator performance.
