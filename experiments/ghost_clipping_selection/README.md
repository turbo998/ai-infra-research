# Per-example clipping and sequence ghost norms / 逐样本裁剪与序列 Ghost Norm

Round 99 checks original linear-model gradients, all-token-pair Gram norms,
flat clipping, stopped weighted replay and physical-partition invariance.
An independent finite-difference derivative checks the chosen update.

```powershell
python -m experiments.ghost_clipping_selection.run
python -m experiments.ghost_clipping_selection.run --check
python -m unittest tests.test_ghost_clipping_selection
```

Three examples each have two tokens and two input/output dimensions.
Loss is half squared error summed across tokens and outputs. The norm joins
weight and bias gradients; coefficients use the pinned `C/(norm+1e-6)`
policy. A fixed synthetic noise vector illustrates one logical release
with expected denominator four, not actual fixture count three.

**This is not private training.** There is no Poisson sampler, accountant,
Gaussian random draw, secure generator, epsilon/delta result or privacy
guarantee. Changing clip policies may require different accounting; a
numerically different update is not automatically a privacy violation.
Mixed precision and repeated/tied module invocations are not assessed.

The separate H5 budget covers selected physical-chunk buffers for one
linear layer: per-example weight gradients, two sequence Gram matrices,
and activations/backprops. A shorter sequence can favor Gram storage;
`T²` can reverse that advantage. This is not peak-HBM or GPU-latency data.
Repeated independent noise at the same final denominator follows a
variance formula, not a validated training implementation.

First-party references at
[`a196a93`](https://github.com/pytorch/opacus/tree/a196a93a8326089242d738e7110fbb625608f114):
`opacus/grad_sample/linear.py`,
`opacus/grad_sample/grad_sample_module_fast_gradient_clipping.py`,
`opacus/utils/fast_gradient_clipping_utils.py` and
`opacus/optimizers/optimizer.py`.
Source supports exact sequence Gram contraction, detached clipping
coefficients, `sigma*C` noise and expected-batch mean scaling; the optimizer
may additionally account for `accumulated_iterations`. Upstream Apache-2.0
rights retained; no implementation copied.

**Synthetic; independently_reproduced=false.** No Opacus/PyTorch or GPU run.
