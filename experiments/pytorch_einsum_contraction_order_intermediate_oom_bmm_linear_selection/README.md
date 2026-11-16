# PyTorch `torch.einsum` Multi-Operand Contraction Order (`32.00 GB` Intermediate OOM) & Implicit `.permute().contiguous()` vs. Explicit `torch.bmm / F.linear` Evaluator

> **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`.

Reproduces the sealed metrics for `docs/components/pytorch-einsum-contraction-order-intermediate-oom-bmm-linear-selection-faq.md`:
- **Tier 1 (Left-to-Right `einsum('bhsd,bhtd,bhtv->bhsv', Q, K, V)` contracting `(Q, K)` first)**: Materializes a `[4, 64, 8192, 8192]` intermediate tensor (**`32.00 GB` HBM**), executes `4,398.0 GFLOPs` (`64x` excess compute), and takes `18.40 ms`.
- **Tier 2 (Permuted `einsum('bshd,bthd,bthv->bshv')` triggering implicit `.permute().contiguous()` around `at::bmm`)**: `6.40 GB` copy/intermediate traffic, `2.85 ms`.
- **Tier 3 (Unfused Eager Reduction `einsum('bshd,bshd->bsh')`)**: Materializes `2.00 GB` elementwise product before `.sum(-1)`, taking `1.95 ms`.
- **Tier 4 (Explicit Right-to-Left Associative `torch.bmm` / `F.linear`: `Q @ (K^T @ V)`)**: Materializes only `[4, 64, 128, 128]` = **`0.008 GB` (`8 MB`, `-31.99 GB` saved)**, `68.7 GFLOPs`, and runs in **`0.12 ms` (`153.3x` faster than Tier 1, `23.8x` faster than Tier 2)**.
