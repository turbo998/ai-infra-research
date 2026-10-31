# SGLang dsv41 SM90 Paged Decode Tile Balancing, Disaggregation First-Refusal Stop & vLLM Fused Sparse-Indexer Top-K (Phase 167)

Deterministic offline model covering:
- `sgl-project/sglang` `2ccbdab65f` (`#47387`): SM90 dsv41 paged decode indexer tile balancing across GPU SMs.
- `sgl-project/sglang` `566dbd2cb3` (`#46898`), `2f6e7f3475` (`#44718`), and `b20fdd2eb3` (`#45876`): Disaggregation decode transfer early stop on first refusal, AMD Mamba slotless HiCache checkpoint storage, and GB300 DFLASH tuned BF16 GEMM dispatch.
- `vllm-project/vllm` `25fc60723c` (`#60618`): Single-launch fused scoring + top-k kernel for sparse-indexer prefill without materializing full logits.
