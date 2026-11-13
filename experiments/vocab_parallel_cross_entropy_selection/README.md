# Vocabulary-parallel cross-entropy and distributed log-sum-exp

Round 94. **Synthetic; independently_reproduced=false; no GPU measurements.**
Distinct from token-chunked linear CE in the existing SFT packing study.

```powershell
python -m experiments.vocab_parallel_cross_entropy_selection.run
python -m experiments.vocab_parallel_cross_entropy_selection.run --check
python -m unittest tests.test_vocab_parallel_cross_entropy_selection
```

The original model partitions output vocabulary columns, reduces a global
maximum, the unique owner's shifted target logit and the exponential denominator,
then computes local softmax gradients. Every gradient coordinate is checked
against independent dense-loss finite differences, including ignored rows and
padded classes. Wrong controls cover AVG instead of SUM, subtracting the target
on every shard, including padded classes and owner-local softmax.

`ignore_index=-100`, padded-class exclusion and explicit no-valid-token status
are **harness policies**, not claims about the upstream kernel API. All-ignored
batches return unknown loss plus an explicit skip-step status, not a fake zero
quality score. Label smoothing, class weights and z-loss are outside scope.

The budget compares FP32 full-logit gathering with a column shard and three
sequential FP32 scalar-statistic reductions per token. Ring startup and wire
bytes are modeled separately. It excludes the output GEMM, hidden-gradient
collectives, optimizer, workspace and backward traffic; it is not a training
step-time or peak-HBM prediction. A single-rank traffic ratio stays null.

The implementation source moved: at the pinned commit, the legacy
`tensor_parallel/cross_entropy.py` is only a compatibility alias. The research
reads `core/ops/loss/cross_entropy.py` rather than inferring behavior from the
old filename. Rendering shares original page/theme helpers with Round 92;
all loss logic and evidence are independent.
