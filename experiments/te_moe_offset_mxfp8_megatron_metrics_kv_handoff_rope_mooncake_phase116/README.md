# TransformerEngine 64-Bit MoE Offset & Grouped MXFP8 / Megatron Disaggregated KV Handoff (`TP > KV-Heads`) & FP32 Vision RoPE / Mooncake Cached DFS & NIXL RAII Model (Phase 116)

Reproducible CPU/offline analytical model (`evidence_mode: own-offline`) for Phase 116 Track C (`priority: 125`):
- TransformerEngine 64-bit (`int64_t`) fused MoE auxiliary loss integer offset guard (`268fdd54`) and grouped MXFP8 `128x128` tile fused requantization (`15dcc848`)
- Megatron-LM Tensor Metrics 4 training observation (`364eef70`), disaggregated KV handoff when `TP > num_kv_heads` (`259f6a7c`), and Vision ViT FP32 RoPE frequency preservation (`3ca52d95`)
- TorchTitan RL `ActorSystem` Monarch config (`f1bb643a`), Mooncake Store cached DFS validated shard path reuse (`d782a599`), LMCache undersized raw-block load rejection (`f17926d0`), Dynamo frontend skip per-token decode without logprobs (`cffc4fa7`), and NIXL Mooncake `postXfer` RAII leak fix (`7bd1d138`)
