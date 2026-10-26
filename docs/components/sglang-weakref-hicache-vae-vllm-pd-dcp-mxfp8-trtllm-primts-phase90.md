# SGLang Scheduler WeakRef、HiCache 空辅助传输跳过、多模态预计算哈希保留、单遍 VAE 拼缝融合，与 vLLM Rust P/D 缓存命中计费、NIXL DCP Interleave、DeepSeek-V4 MXFP8/MXFP4 及 TensorRT-LLM PrimTS 原生稀疏 MLA 解析（Phase 90）

> **English Version**: [sglang-weakref-hicache-vae-vllm-pd-dcp-mxfp8-trtllm-primts-phase90.en.md](sglang-weakref-hicache-vae-vllm-pd-dcp-mxfp8-trtllm-primts-phase90.en.md)  
> **证据等级**：`implementation`（上游固定 commit 源码审查）+ `own-offline`（本项目 Python 3.12 标准库确定性建模 `experiments/sglang_weakref_hicache_vae_vllm_pd_dcp_mxfp8_trtllm_primts_phase90/run.py`）。未运行真实 GPU/RDMA 硬件 benchmark。

## 1. 一手源码证据与关键实现路径

1. **SGLang Scheduler `Req.__req__` 弱引用防内存泄漏**（`sgl-project/sglang` PR `#45923`, commit `3a62cdfb3d`）：
   - 自定义 logit processor（如 `Qwen3ThinkingBudgetLogitProcessor`）通过 `custom_params["__req__"]` 访问当前 `Req`。此前强自引用形成引用环，导致被取消/中止请求在 GC 周期前滞留显存与主机堆；改为 `weakref.proxy(self)` 后支持即时引用计数回收（模型中 4096 个中止请求可立即释放 $256\text{ MiB}$ 闭包缓冲）。
2. **SGLang HiCache 跳过空辅助池 L2 传输**（`sgl-project/sglang` PR `#45673`, commit `1596e88eda`）：
   - 在 `HybridCacheController._l2_transfers` 中，当 `pool_transfer.host_indices.numel() == 0` 且 `device_indices.numel() == 0` 时（例如混合架构中仅 `KV` 池有增量而 `SWA` / `INDEXER` 池为空），直接跳过空辅助池描述符下发，消除零长度 DMA/RDMA 调度开销。
3. **SGLang 多模态预计算 Embedding 保留调用方哈希与 Offsets**（`sgl-project/sglang` PR `#44160`, commit `ae579c5e00`）：
   - 在 `BaseMultimodalProcessor.process_and_combine_mm_data` 处理 `PRECOMPUTED_EMBEDDING` 输入时，显式保留调用方传入的 `offsets` 与受信媒体身份哈希，使预计算多模态特征能命中 RadixKV 前缀树复用。
4. **SGLang Diffusion ParallelTiledVAE 单遍拼缝融合（One-Pass Seam Blending）**（`sgl-project/sglang` PR `#46240`, commit `d72664d903`）：
   - 在 `vaes/common.py` 中新增静态 `_blend(a, b, blend_extent, dim)`，将相邻 VAE tile 的重叠区域在一次张量切片遍历中完成线性交叉淡入淡出（cross-fade），将拼缝融合的读写遍数减半（节省 $50\%$ 拼缝 I/O）。
5. **vLLM Rust Frontend 使用 Prefiller 缓存命中统计 P/D Usage**（`vllm-project/vllm` PR `#58885`, commit `c36154742e`）：
   - 在 P/D 分离架构下，Decode 节点本地前缀缓存命中为 0，真实前缀命中发生在远程 Prefiller。Rust 前端在 `GenerateOutputStream` 中记录 `remote_prefill_cached_tokens` 并写入最终 `Usage.prompt_tokens_details.cached_tokens`，修复 P/D 分离场景下缓存命中率被少报为 0 的计费偏差。
6. **vLLM NIXL DCP Interleave Size 在 Profiling 前定型**（`vllm-project/vllm` PR `#62330`, commit `3e75ccba29`）：
   - 在 KV cache 规划与显存 profiling 阶段之前即完成 Distributed Context Parallel (DCP) 的 NIXL `dcp_interleave_size` 对齐（对齐至 `block_size=16`），防止 profiling 使用未对齐粒度导致 KV 页表预算失配。
7. **vLLM DeepSeek-V4 Hopper MXFP8->Block-FP8 回退与 ROCm CDNA4 默认 MXFP4 Indexer K Cache**（`vllm-project/vllm` PRs `#61875` `fcb787055e`, `#63669` `d0d681e40d`）：
   - 在 Hopper (`SM90`) 上声明 DeepSeek-V4 MXFP8 MoE 激活并以 block-FP8 (`group_size=128`) 执行；
   - 在 AMD CDNA4 (`MI350/MI355`) 上将 DeepSeek-V4 稀疏注意力 Indexer K cache 默认设为 `mxfp4`（每元素 $0.5 + 1/32 = 0.53125\text{ B}$，相比 FP8 节省 $46.88\%$ Indexer K cache 显存，释放约 $9\%$ 全局 KV cache 容量）。
8. **TensorRT-LLM `PyTorchModelEngine` 命名阶段拆分与 SM100 默认启用 PrimTS Native Sparse MLA FMHA**（`NVIDIA/TensorRT-LLM` PRs `#20851` `011316470c`, `#20209` `178ac92fed`）：
   - 将 `PyTorchModelEngine` 构造拆分为模型加载、KV/资源初始化、CUDA Graph 预热捕获三个独立阶段；
   - 集成 `flashinfer-prims-ts` 原生稀疏 MLA FMHA 并在 SM100 上为 DeepSeek-V4 生成默认开启（$L=32768, \text{top\_k}=2048$ 时减少 $16\times$ KV 读取）。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：在 MI350 上运行 DeepSeek-V4 Indexer K cache，每元素从 FP8 ($1.0\text{ B}$) 切换为 MXFP4 ($0.5\text{ B}$ 数据 + 每 32 元素 $1\text{ B}$ E8M0 缩放)，显存节省百分比是多少？
  - 答：$\frac{1.0 - 0.53125}{1.0} = 46.875\% \approx 46.88\%$。
- **可证伪假设**：在 P/D 分离网关中若仅读取 Decode 侧 `num_cached_tokens` 而不透传 `remote_prefill_cached_tokens`，所有远程 Prefill 请求的 `cached_tokens` 指标恒为 0。
