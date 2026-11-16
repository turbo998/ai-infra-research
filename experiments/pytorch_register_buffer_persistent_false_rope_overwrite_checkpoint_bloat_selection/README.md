# PyTorch `register_buffer(persistent=True vs False)` RoPE Context-Extension Overwrite Bug & `18.40 GB` Checkpoint Bloat Evaluator

> **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`.

Reproduces the sealed metrics for `docs/components/pytorch-register-buffer-persistent-false-rope-overwrite-checkpoint-bloat-selection-faq.md`:
- **Tier 1 (Default `register_buffer("inv_freq", inv_freq, persistent=True)`)**: `load_state_dict()` silently overwrites the newly initialized `rope_theta = 500,000` buffer with the stale `10,000` buffer from the base checkpoint, causing `64K` perplexity to explode from `5.42` to **`184.60` (`34.1x` worse)**.
- **Tier 2 (Per-Layer `cos_cached / sin_cached / causal_mask` with `persistent=True`)**: Bloats every saved checkpoint by **`+18.40 GB`** across 80 layers and fails `load_state_dict(strict=True)` when sequence length changes.
- **Tier 3 (Plain Python Tensor Attribute `self.inv_freq = inv_freq`)**: Ignored by `model.to("cuda")` / `FSDP`, causing a `100%` device mismatch crash (`cpu` vs `cuda:0`).
- **Tier 4 (`register_buffer("inv_freq", inv_freq, persistent=False)` + Model-Level Shared `RotaryEmbedding`)**: `0.00 GB` checkpoint bloat (`-18.40 GB` saved), immune to `load_state_dict()` overwrites (`64K` PPL = `5.42`), and `0.0%` device-move crash.
