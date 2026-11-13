# Contrastive global negatives and GradCache replay

Round 93. **Synthetic, independently_reproduced=false, no trained model or GPU timings.**

```powershell
python -m experiments.contrastive_gradcache_selection.run
python -m experiments.contrastive_gradcache_selection.run --check
python -m unittest tests.test_contrastive_gradcache_selection
```

The original toy uses four query/document pairs, a shared scalar parameter,
two-dimensional `mask * tanh(theta * input)` embeddings and one-way dot-product
InfoNCE. Stable softmax derivatives are computed analytically; central finite
differences provide an independent reference. The modeled two-worker detached
gather drops candidate-gradient contributions from remote query rows while
retaining the local candidate slice. It is not a simulation of every possible
gather or DDP implementation.

The exact replay path contracts cached embedding gradients with the same encoder
Jacobian in chunks. A changed-dropout control uses different masks for replay.
The local-negative control averages separate two-pair objectives: gradient
accumulation is not a union of their softmax denominators.

The separate H5 budget uses explicitly assumed activation bytes per pair and
forward/backward cost. It counts conservatively replicated embeddings plus FP32
embedding gradients and FP32 score tiles. It excludes model/optimizer memory,
softmax temporaries, allocator fragmentation, communication and backend overhead;
it is **not a peak-HBM guarantee**. Dense and cached budgets are explanatory
proxies, not complete allocator models.

Rendering shares only the original static page/theme utilities with Round 92;
gradient logic and all evidence are independent. Sources, protocol and limitations
are in the [English FAQ](../../docs/components/contrastive-embedding-global-negatives-gradcache-selection-faq.en.md).
