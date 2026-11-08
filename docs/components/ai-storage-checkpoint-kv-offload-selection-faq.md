# [导航索引] 分布式训练 Checkpoint 存储与长上下文 KV Cache 分层卸载 FAQ（已拆分为两个独立单专题）

> 🌐 **Language / 语言**: **中文 (Chinese)** | **[English (`ai-storage-checkpoint-kv-offload-selection-faq.en.md`)](ai-storage-checkpoint-kv-offload-selection-faq.en.md)**
> 🧭 **单专题拆分说明**：训练 Checkpoint 容灾持久化和推理长上下文 KV Cache 卸载属于不同生命周期的独立架构决策，已分别拆分为以下 **2 篇独立单决策六段式 FAQ**：

1. **Round 3 独立专题：分布式大模型训练 Checkpoint 存储架构选型（`同步阻塞 POSIX` vs. `同步直通 GDS` vs. `两阶段异步 PyTorch DCP`）**
   - 中文：[`docs/components/training-checkpoint-architecture-selection-faq.md`](training-checkpoint-architecture-selection-faq.md) | English：[`docs/components/training-checkpoint-architecture-selection-faq.en.md`](training-checkpoint-architecture-selection-faq.en.md)
2. **Round 4 独立专题：长上下文推理 KV Cache 分层卸载架构选型（`全量重算` vs. `本机 Pinned DRAM` vs. `跨机共享 RDMA NVMe + GDS 池`）**
   - 中文：[`docs/components/kv-cache-tiered-offload-selection-faq.md`](kv-cache-tiered-offload-selection-faq.md) | English：[`docs/components/kv-cache-tiered-offload-selection-faq.en.md`](kv-cache-tiered-offload-selection-faq.en.md)
3. **关联算例与 7-Card 报告**：[`experiments/ai_storage_checkpoint_kv_offload/results.json`](../../experiments/ai_storage_checkpoint_kv_offload/results.json) · [`docs/reports/ai-storage-checkpoint-kv-offload-selection-cards.html`](../reports/ai-storage-checkpoint-kv-offload-selection-cards.html)
