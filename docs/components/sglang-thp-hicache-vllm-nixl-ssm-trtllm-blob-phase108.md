# SGLang 并行 THP HiCache 分配与 DP-Attention 填充、vLLM 混合 SSM NIXL 区域过滤及 TensorRT-LLM Blob 后端分析 (Phase 108 · Priority 117)

> **English Version**: [sglang-thp-hicache-vllm-nixl-ssm-trtllm-blob-phase108.en.md](sglang-thp-hicache-vllm-nixl-ssm-trtllm-blob-phase108.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/sglang_thp_hicache_vllm_nixl_ssm_trtllm_blob_phase108/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`sgl-project/sglang@75468e4b4f` (`#43978`: `python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py`, `environ.py`)**：
   - **HiCache 主机 KV 池多线程透明大页（THP）并行预缺页分配**：新增 `SGLANG_HUGEPAGE_SIZE`（支持 `2MB` / `1GB`）与 `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS`（工作线程数）。对于 `256 GiB` 主机 KV 缓存池，采用 `2 MiB` 大页将缺页中断次数从 `4 KiB` 页的 `67,108,864` 次降至 **`131,072` 次（减少 `512x`）**，配合 16 线程并行 prefault 将池初始化耗时从 `14.20 s` 压缩至 **`0.94 s`（`15.11x` 启动加速）**。
2. **`sgl-project/sglang@58eb1740d2` (`#45796`: `python/sglang/srt/model_executor/forward_batch_info.py`) 与 `5025cbf7d4` (`#46658`: `elementwise.py`)**：
   - **DP-Attention 异地（Out-of-Place）宿主列表填充与符号形状安全护栏**：在 DP-Attention 填充阶段，预先规划的 DeepSeek Sparse Attention（DSA）元数据持有 `extend_seq_lens_cpu` 与 `extend_logprob_start_lens_cpu` 的引用。`#45796` 将原位 `.extend()` 改为异地拼接重绑定，避免将 4 个原始请求的 DSA 元数据列表污染膨胀为 8 行（**消除 `4` 行幻影元数据越界**）；`#46658` 在 `sigmoid-mul` 启动器中以 `guard_or_false` / `guard_or_true` 替代数据依赖的 token 数判断，防止 `torch.compile` 符号形状重编译。
3. **`vllm-project/vllm@4ddc7aa40f` (`#62317`: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`)**：
   - **混合 Attention + SSM (Mamba) 模型 NIXL KV Connector 按组区域过滤**：在混合内存架构（HMA）下为每个 KV Cache Group 维护专属 `_group_region_indices`，使 NIXL RDMA 读取仅扫描所属组的内存描述符（`48` 个 Attention 区域 + `16` 个 SSM 状态区域 = `64` 个描述符，而非未过滤前的 `106` 个跨组描述符），将单请求分离式 KV 传输载荷从 `159.0 MiB` 降至 **`96.0 MiB`（`1.6562x` 带宽压缩，节省 `39.62%` RDMA 流量）**。
4. **`vllm-project/vllm@f03b704da2` (`#62133`)、`08d94f0176` (`#63920`)、`08b989da99` (`#63063`) 与 `ef2f382fd1` (`#63894`)**：
   - `#62133` 修复 SM110 上的 OAI Triton MXFP4 MoE 设备能力检查与分块约束；`#63920` 为 Muse Spark NVFP4 MegaMoE 输入准备核 `_norm_quant_megamoe_nvfp4_kernel` 接入 `triton_kernel_dispatcher_with_warmup` 预热以消除首请求 JIT 编译抖动；`#63063` 在 DeepSeek V4.1 DSpark 中使用 `MergedColumnParallelLinear` 纵向堆叠上下文 WKV 投影；`#63894` 在 `_fused_moe_lora_kernel` 中引入 `FP32_ADD_INPUTS: tl.constexpr`，消除 LoRA 展开阶段的二次舍入误差。
5. **`NVIDIA/TensorRT-LLM@b351c407bb` (`#20884`: `tensorrt_llm/_torch/disaggregation/blob/`) 与 `c554d4486d` (`#20227`: `model_config.py`)**：
   - `#20884` 新增面向分离式 KV 共享的 `BlobBackend`（`BlobBackend -> StoreDriver -> byte store SDK`），支持批量内存注册（`batch_size = 64` 将 `512` 个 KV 块的注册耗时从 `25.60 ms` 降至 **`2.92 ms`，提速 `8.77x`**）与有界异步队列准入（`max_queue_depth = 128`）；`#20227` 在 `FORCE_DETERMINISTIC=1` 或 `FORCE_MOE_KERNEL_DETERMINISTIC=1` 时自动设置 `moe_disable_finalize_fusion = True`，消除融合专家归约的跨运行原子累加顺序漂移（`max_abs_diff = 0.0`）。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：对于 `256 GiB` 的 SGLang HiCache 主机内存池，对比 `4 KiB` 基础页与 `2 MiB` 透明大页（THP）所需的页表项/缺页中断次数，并计算为何 `2 MiB` 大页可将内核缺页处理次数精确压缩 $2,097,152 / 4,096 = 512\times$。
- **可证伪假设**：在混合 Attention + Mamba SSM 模型中，若 NIXL KV Connector 对每个 KV Cache Group 遍历全部已注册内存区域而非 `_group_region_indices`，Attention 组会重复发起针对 SSM 状态的无效远端 RDMA 读取（被 `vllm #62317` 单元测试与几何描述符检查证伪并修复）。
