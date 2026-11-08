# [Navigation Portal] Distributed Training Checkpoint & Long-Context KV Cache Tiered Offload FAQs (Split into Two Single-Decision Guides)

> 🌐 **Language / 语言**: **[中文 (`ai-storage-checkpoint-kv-offload-selection-faq.md`)](ai-storage-checkpoint-kv-offload-selection-faq.md)** | **English**
> 🧭 **Single-Decision Split Notice**: Training checkpoint persistence and inference long-context KV cache offloading are distinct architectural decisions and are maintained as **two standalone single-decision 6-part FAQ guides**:

1. **Round 3 Standalone FAQ: Distributed Training Checkpoint Architecture Selection (`Synchronous Blocking POSIX` vs. `Synchronous Direct GDS` vs. `Two-Phase Asynchronous PyTorch DCP`)**
   - English: [`docs/components/training-checkpoint-architecture-selection-faq.en.md`](training-checkpoint-architecture-selection-faq.en.md) | 中文: [`docs/components/training-checkpoint-architecture-selection-faq.md`](training-checkpoint-architecture-selection-faq.md)
2. **Round 4 Standalone FAQ: Long-Context Inference KV Cache Tiered Offload Selection (`Full Prefill Recompute` vs. `Host Pinned DRAM` vs. `Shared RDMA NVMe + GDS Pool`)**
   - English: [`docs/components/kv-cache-tiered-offload-selection-faq.en.md`](kv-cache-tiered-offload-selection-faq.en.md) | 中文: [`docs/components/kv-cache-tiered-offload-selection-faq.md`](kv-cache-tiered-offload-selection-faq.md)
3. **Sealed Evaluator & 7-Card Deck**: [`experiments/ai_storage_checkpoint_kv_offload/results.json`](../../experiments/ai_storage_checkpoint_kv_offload/results.json) · [`docs/reports/ai-storage-checkpoint-kv-offload-selection-cards.html`](../reports/ai-storage-checkpoint-kv-offload-selection-cards.html)
