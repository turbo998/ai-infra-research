# FlashInfer SM90 GDN Scratch Bounds, SGLang AMD DSV4.1 Layer-20 Replay Skip, and TensorRT-LLM CUTLASS MoE / MLA Gates

Deterministic standard-library verification for:
- `flashinfer-ai/flashinfer@afbe429cc26216343f00c17fc9aa23fa5d8449ed` (`#6849`)
- `sgl-project/sglang@82ac83a8d0ab50ea7c9fe8e5b5934c2fa5a224bf` (`#45339`)
- `NVIDIA/TensorRT-LLM@fe92fb923f12f98cfafc180bd9496126d2d9ef9e` (`#19753`) & `4c0197d1e4410eb005748224ffd4c2ca5f0c7b44` (`#20652`)

Run:
```bash
python experiments/flashinfer_gdn_sm90_sglang_dsv41_trtllm_moe/run.py --check
```
