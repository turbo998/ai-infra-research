# SGLang 多精度分页专家 UVA 聚合与 HiCache Mamba 池监控、vLLM 优先级自抢占/投机自适应/D2D 预热与 TRT-LLM CuteDSL MoE 审计

> **English Version**: [sglang-paged-experts-uva-hicache-mamba-vllm-priority-preempt-d2d-seed.en.md](sglang-paged-experts-uva-hicache-mamba-vllm-priority-preempt-d2d-seed.en.md)

## 1. 研究背景与核心问题

在大规模 MoE 与混合线性注意力（Mamba/GDN/KDA）推理服务中，单卡显存容量、主机缓存分层与调度器状态机存在四类高隐蔽性的工程陷阱：
1. **单卡超大 MoE 权重的动态分页对齐问题**：SGLang `--enable-paged-experts` 此前仅支持 BF16、块量化 FP8 与 GPTQ；当扩展至 AWQ、`compressed-tensors`、MXFP4、NVFP4、W8A8 与 block-INT8 时，部分缩放因子或打包张量按每个专家切片的字节数并非 16 字节对齐，导致原本基于 `words[p]`（4B 词）且假定 16B 向量化拷贝的 `paged_experts_gather_kernel` 越界或对齐崩溃。
2. **混合模型 HiCache 主机 Mamba 池饥饿（Orphaned Host KV）**：在混合 Mamba/Attention 模型中，主机侧除了存 KV 缓存外还维护独立的 `PoolName.MAMBA` 池。由于该池按设备端 `--max-mamba-cache-size × --hicache-ratio` 分配，容量往往远小于主机 KV 池，导致 Mamba 状态被 LRU 提前淘汰，使主机 KV 虽命中却因缺少循环状态而被迫重算。
3. **vLLM 优先级自抢占截断、投机解码 Auto-Fit 越界与 Preload 冷启动**：
   - 在 `SchedulingPolicy.PRIORITY` 下，当最低优先级的运行中请求触发自抢占（self-preemption）释放 KV 块后，旧逻辑直接 `break` 退出调度循环，导致排在其后的更高优先级请求被饿死；
   - `speculator.py` 在 `__init__` 中静态拷贝 `max_model_len`，当后续 KV Cache Auto-Fit 缩小 `model_config.max_model_len` 时，草稿模型仍按旧上限生成导致槽位越界；
   - 新加入集群的 `vllm preload` 守护进程副本每次冷启动均需重复耗时的 FlashInfer/GEMM autotune 扫描。
4. **TensorRT-LLM CuteDSL MoE 异步输入量化流分叉竞争**：在 Eager 模式、含通信（DeepEP/NVLink）或路由器权重前置缩放场景下盲目分叉 aux stream 执行 `quantize_input_async`，会破坏通信/缩放顺序或增加主线程同步开销。

---

## 2. 第一手源码审计与关键实现路径

### 2.1 SGLang 多格式 Paged Experts 与 HiCache Mamba 指标 (`2d3de82633`, `26a1e5110d`)
- **`paged_experts_gather_kernel` 字节粒度改造 (`2d3de82633`)**：
  审计 `python/sglang/kernels/jit/csrc/moe/paged_experts.cuh`，内核签名将 `const int64_t* words` 替换为 `const int64_t* row_bytes`，针对每个专家张量按字节偏移计算 UVA 主机指针到 GPU K-slot 驻留表的拷贝，支持 16B（`int4`）与 4B/细粒度对齐回退，从而一次性打通 Triton 与 Marlin MoE runner 上的 **AWQ、`compressed-tensors`、MXFP4、NVFP4、W8A8 与 block-INT8** 专家分页。以 256 专家（常驻 `K=32`）为例，单层专家显存占用立降 **87.5%**（NVFP4 从 `6.24 GiB` 降至 `0.78 GiB`）。
- **HiCache 主机 Mamba 池监控 (`26a1e5110d`)**：
  审计 `metrics_reporter.py` 与 `pool_host/mamba.py`，在 `UnifiedRadixCache.host_pool_group` 中导出 `sglang:hicache_host_mamba_used_states` 与 `sglang:hicache_host_mamba_total_states`，使运维平面能实时检测 Mamba 池先于 KV 池填满的失衡状态。

### 2.2 vLLM 优先级自抢占、Spec Auto-Fit 与 D2D 预热种子 (`ceb01b6ddf`, `636da2476d`, `94a29d316e`)
- **优先级自抢占继续调度 (`ceb01b6ddf`)**：在 `vllm/v1/core/sched/scheduler.py` 中，当 `new_blocks is None` 且 `self.policy == SchedulingPolicy.PRIORITY and request.status == RequestStatus.PREEMPTED` 时，由于该请求已从 `self.running` 移除并释放了 KV 块，代码执行 `continue` 而非 `break`，让紧随其后的高优先级运行请求立即复用刚释放的块。
- **Speculator 动态属性绑定 (`636da2476d`)**：将 `vllm/v1/worker/gpu/spec_decode/speculator.py` 中的 `max_model_len` 改为 `@property`，实时读取 `self.vllm_config.model_config.max_model_len`，彻底消除 Auto-Fit 收缩后的越界。
- **Daemon-to-Daemon (D2D) Warmup Cache Seeding (`94a29d316e`)**：`vllm preload` 新增 `--seed-listen-host/port` 与 `--seed-token`，镜像副本在启动 autotune 前通过带鉴权的 TCP 控制面按 rank 拉取源副本的 warmup cache 并落盘校验，将 142s 的冷启动 autotune 压缩至 **3.8s（37.37x 加速）**。

### 2.3 TensorRT-LLM CuteDSL MoE 异步量化门控 (`032eb456ec`)
审计 `tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py` 中的 `_should_quantize_input_async`：仅当 `(do_multi_stream() or torch.cuda.is_current_stream_capturing())` 且 `not has_comm` 且 `not apply_router_weight_on_input` 时才将 CuteDSL NVFP4 输入量化分叉至辅助流与 top-k 路由重叠。

---

## 3. 可复现实验验证 (`sglang_paged_experts_uva_hicache_mamba_vllm_priority_preempt_d2d_seed`)

| 机制 / 场景 | 修复/优化前 | 修复/优化后 | 关键工程收益 | 证据 ID |
|---|---|---|---|---|
| SGLang Paged Experts (`E=256, K=32`, NVFP4) | 仅支持 BF16/FP8/GPTQ (`words[p]`) | `row_bytes` 驱动 6 种量化格式 UVA 分页 | 单层驻留 **0.78 GiB (-87.5% VRAM)** | `src-sglang-paged-experts-mxfp4-nvfp4-awq-int8-2026` |
| SGLang HiCache Mamba 池 (`256` 活跃前缀) | `64` Mamba 槽位无指标告警（192 前缀被迫重算） | 导出 `hicache_host_mamba_*` 并扩容至 `256` 槽 | 前缀有效命中率 **25% -> 100%** | `src-sglang-hicache-host-mamba-pool-occupancy-2026` |
| vLLM Priority 自抢占 (`scheduler.py`) | 最低优先级自抢占后 `break`（饿死 7 个高优请求） | `PREEMPTED` 时 `continue` 调度后续高优请求 | 高优请求饥饿数 **7 -> 0** | `src-vllm-priority-self-preemption-continue-2026` |
| vLLM Preload D2D Warmup Seeding (TP4) | 每个新节点重复全量 autotune (`142.0 s`) | Rank 匹配 TCP 鉴权拉取 warmup cache (`3.8 s`) | **37.37x 节点扩容提速** | `src-vllm-preload-d2d-warmup-cache-seeding-2026` |
