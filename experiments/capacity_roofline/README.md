# Capacity and Roofline (own-offline)

This synthetic, standard-library experiment is arithmetic, **not a hardware
benchmark**. It executes no model, measures no timings, and establishes no
quality, throughput, or product capability.

From the repository root:

```powershell
python -m experiments.capacity_roofline.run
python -m experiments.capacity_roofline.run --check
```

The first command regenerates [results.json](results.json) and
[results.csv](results.csv) from [input.json](input.json). The second compares
their exact UTF-8 bytes without modifying them. Direct execution of `run.py`
also works. `--input` accepts another synthetic input but outputs always go to
this experiment directory.

## Contracts

- Weight payload = `ceil(parameters * bits_per_parameter / 8)` bytes. Integer
  packing does not include scales, zero points, metadata, or alignment.
- Conventional MHA/GQA logical KV =
  `2 * layers * kv_heads * head_dim * sequence * batch * bytes_per_element`.
  Dimensions and element bytes must be positive integers (not booleans or
  fractional bytes). MLA and hybrid architectures are explicitly unsupported.
- Per-sequence allocated tokens =
  `ceil(sequence / token_block_size) * token_block_size`. Accounted capacity uses
  **allocated** KV, not logical KV, plus separately stated workspace and allocator
  reserve. Allocation assumes equally sized sequences in the batch and no shared
  prefix reuse or fragmentation beyond token-block padding.
- A null capacity produces null fit and headroom assessments. Otherwise headroom
  is `capacity - accounted_total` (negative means over capacity). A true fit only
  concerns accounted bytes, never all real runtime memory.
- Arithmetic intensity = `flops / bytes_moved`; ridge = `peak / bandwidth`;
  performance upper bound = `min(peak, bandwidth * intensity)`; elapsed-time
  lower bound = `max(flops / peak, bytes_moved / bandwidth)`. All scalar inputs
  must be finite and positive. These are ideal bounds, **not timings**.
- GB = `10**9` bytes; GiB = `2**30` bytes. FLOP means one floating-point operation;
  input peak must use the same operation-count convention as input work.

## Independent synthetic reference values

24 layers, 4 KV heads, head dimension 128, 8192 tokens, **batch 2**, and BF16
(2 bytes/element) yield **805,306,368 KV bytes = 0.75 GiB**. A 16-token block
adds no padding for this sequence. One billion 16-bit parameters require
**2,000,000,000 weight-payload bytes**. Synthetic workspace and allocator reserve
are **268,435,456 bytes each**, giving **3,342,177,280 accounted bytes** and
**952,790,016 bytes of accounted headroom at 4 GiB**. These supplied overheads
are not measurements and do not enumerate every real runtime allocation.

The same dimensions with **batch 1** yield **402,653,184 bytes = 0.375 GiB**;
tests independently assert both cases. Assigning the 0.375 GiB result to
batch 2/BF16 would violate the stated formula.

Synthetic work `1e12` FLOP, traffic `1e10` bytes, peak `1e14` FLOP/s, and memory
bandwidth `1e12` bytes/s give intensity **100 FLOP/byte**, ridge **100 FLOP/byte**,
upper bound **1e14 FLOP/s**, and lower bound **0.01 s**.

Roofline omits launch/communication latency, cache effects, utilization, and
overlap constraints. A network/ring model is deliberately deferred; no duplex
aggregate rate is converted to one-way elapsed time.
