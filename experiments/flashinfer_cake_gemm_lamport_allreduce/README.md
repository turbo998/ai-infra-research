# FlashInfer Cake SM100/SM103 Fused GEMM + MNNVL Lamport All-Reduce & tcgen05 Sync (`flashinfer_cake_gemm_lamport_allreduce`)

Deterministic Python 3.12 standard-library verification of FlashInfer's Blackwell (`SM100` / `SM103`) Cake fused tensor-parallel GEMM + MNNVL Lamport all-reduce (`e563bf9867`), `tcgen05.wait::ld` TMEM drain before dealloc (`3bf6d32022`), and 2-CTA (`cta_group::2`) producer-tail stage release drain (`b815220c91`).

## Run

```bash
python experiments/flashinfer_cake_gemm_lamport_allreduce/run.py --check
```
