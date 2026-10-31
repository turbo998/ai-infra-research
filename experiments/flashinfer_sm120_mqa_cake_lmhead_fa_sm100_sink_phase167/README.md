# FlashInfer SM120 Paged FP8 MQA Logits, Cake SM100a LM-Head Loss & FA SM100 Softmax/Sink (Phase 167)

Deterministic offline model covering:
- `flashinfer-ai/flashinfer` `e5eb9a5019` (`#8136`): SM120a RTX 5090 paged FP8 MQA logits native-atom routing for `h=64, next_n=5/6`.
- `flashinfer-ai/flashinfer` `b5be99dd30` (`#7858`): Cake chunked LM-head + loss `sm_100a` `expect-tx` arrive ordering under `nvcc` 12.9/13.0.
- `Dao-AILab/flash-attention` `117d189e87` (`#2869`) & `62892fe4a7` (`#2768`): CuTe SM100 non-causal `hdim=128` softmax spill elimination and learnable attention sink for SM100 FlashMLA.
