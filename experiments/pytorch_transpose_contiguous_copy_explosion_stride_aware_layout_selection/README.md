# PyTorch `.transpose().contiguous()` & `.reshape()` Silent Copy Explosion vs. Native `[B, S, H, D]` Stride-Aware Kernel Dispatch Evaluator

> **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`.

Reproduces the sealed metrics for `docs/components/pytorch-transpose-contiguous-copy-explosion-stride-aware-layout-selection-faq.md`:
- **Tier 1 (Explicit `.transpose(1, 2).contiguous()` + `torch.cat` `rotate_half` RoPE)**: `24.00 GB` pure `aten::copy_` HBM traffic per layer, `+4.00 GB` extra activation HBM per layer, `7.16 ms/layer` overhead.
- **Tier 2 (Silent `.reshape()` Copy on Non-Contiguous Transposed Views)**: `12.00 GB` hidden copy traffic per layer, `+2.00 GB` extra HBM, `3.58 ms/layer`.
- **Tier 3 (Inner-Dimension `k.transpose(-2, -1)` Breaking `stride(-1) == 1`)**: Forces internal `.contiguous()` inside kernel wrappers (`6.00 GB` copy traffic, `1.85 ms/layer`).
- **Tier 4 (Native `[B, S, H, D]` / Stride-Aware Zero-Copy Views with `stride(-1) == 1` + In-Place Fused RoPE)**: `0.00 GB` copy traffic, `0.00 GB` extra HBM (`-4.00 GB` saved/layer), `0.18 ms/layer` (`39.8x` faster than Tier 1).
