# Streaming codec state: aligned ingress, overlap-add and slot lifecycle

Round 95. **Synthetic; independently_reproduced=false; no GPU or audio-quality
measurement.** This is an original scalar state-machine model, not Mimi model
inference or a clone of its implementation.

```powershell
python -m experiments.streaming_codec_state_selection.run
python -m experiments.streaming_codec_state_selection.run --check
python -m unittest tests.test_streaming_codec_state_selection
```

Independent offline formulas validate streamed causal and transposed
convolution. The pinned source requires positive chunks divisible by stride.
The original ingress adapter deliberately accepts arbitrary packet lengths,
buffers incomplete frames and pads only once at EOS. This adapter policy is
not attributed to upstream. An alternative reject-at-EOS policy fails explicitly.

The chosen alignment keeps effective-kernel minus stride history, not the
common kernel-minus-one convention. Transpose carry excludes the bias that
the next chunk already adds, and its final K-S tail is trimmed rather than
flushed. Slot history and pending ingress must be reset together on reuse.

Failure controls include resetting history on every chunk, dropping transport
remainders, counting transpose bias twice and reusing a stale slot. Output
length mismatch yields unknown numeric error, not a zip-truncated zero.

The calculator counts persistent buffers for one illustrative layer pair with
matching input/output channel counts, plus bounded pending ingress. Collection
time is a stage-rate bound, not network/GPU/service latency. It excludes other
codec layers, RVQ, transformer caches, activations, weights and allocator costs.
