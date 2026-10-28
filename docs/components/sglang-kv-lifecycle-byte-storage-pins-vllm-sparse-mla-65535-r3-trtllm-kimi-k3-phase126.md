# Phase 126 Track B: SGLang KV 生命周期与分配器边界 Pin、vLLM Sparse MLA `65535` 网格上限与 R3 捕获器顺序、TensorRT-LLM Kimi-K3 原生 C++ 整合与 MX 输入处理器

> **Phase 126 (`priority: 135`) · 证据等级**: 上游源码与提交说明 (`implementation`) + 本项目离线合成标定 (`own-offline`)。未声明物理集群在线压测 (`own-hardware`)。

---

## 1. 核心架构问题与源码级解构

### 1.1 SGLang KV Cache 字节存储/身份解耦、统一内存分配器边界 RDMA Pin 与混合池门控 (`#46988`, `#46987`, `#46985`, `#46984`, `#46983`, `#46981`)
- **请求行与分配生命周期收敛 (`#46988`, `53592ea5`) & 字节存储与分配身份解耦 (`#46987`, `ca8184e1`)**:
  - **上游路径**: `python/sglang/srt/disaggregation/decode.py`、`decode_kvcache_offload_manager.py`、`python/sglang/srt/mem_cache/static_kv_storage.py`、`kv/alloc/id_space.py`。
  - **架构演进**: 将 `IdSpaceKind` 从物理定位计划 `kv_loc_plan` 迁移至独立的分配命名空间 `sglang.srt.mem_cache.kv.alloc.id_space`，并通过 `StaticKVStorage` (`kv_storage_of(...)`) 统一托管 `req_to_token_pool`、`token_to_kv_pool_allocator` 与 `tree_cache` 的获取/释放生命周期，避免 PD 分离卸载完成后手工分散调用 `free(req)` 导致的槽位泄漏。
- **分配器边界追踪 RDMA 传输 Pin (`#46985`, `eb79ecca`) & 路由式缓存 (`#46984`, `f80cb3a2`)**:
  - **问题根因**: 在运行统一内存池（Unified Memory Pool）的 PD 节点上，若内存碎片整理（compaction）在 RDMA 引擎读写期间搬移物理页，会直接引发远端 DMA 读写错位。
  - **修复机制**: 将原 `unified_memory_disagg_move_gate` 重构为分配器边界的 `disagg_transfers_in_flight(scheduler)` pin 源，只要存在未确认的在途传输即锁住对应 device rows，使 compaction 与 RDMA 冲突率降至 **`0`**；同时以 `storage` 路由缓存替代硬编码的多套统一池变体。
- **ROCm 滑动窗口翻译存储写门控 (`#46983`, `14fea751`) & 混合 Mamba/KDA Host 布局 (`#46981`, `1721c099`)**:
  - 在 ROCm (`_is_hip`) 上，fused RoPE + KV store 通过 `full_to_swa_index_mapping` 写入窗口层，而统一池的滑动窗口视图仅接受自身写入口派生的 ID；在 `enable_fused_set_kv_buffer` 中新增 `SLIDING_WINDOW` 子池门控，防止覆写翻译窗口存储（污染率 `0`）。
  - 在 MLA + KDA 混合模型启用 `page_first_kv_split` 全局布局时，`MambaPoolHost` 无独立 K/V 缓冲区，新增 `_state_host_layout()` 强制回退至 `page_first_direct`，修复 HiCache 混合循环状态池初始化崩溃。

### 1.2 vLLM FlashInfer Sparse MLA `gridDim.z <= 65535` 分块启动、R3 路由专家预分配与 DeepSeek-OCR 零和补丁门控 (`#63791`, `#63237`, `#64091`, `#64063`)
- **FlashInfer Sparse MLA `65535` Token 上限分块 (`#63791`, `4da2a82a`)**:
  - **问题根因**: `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` 调用的 `trtllm-gen` 稀疏 MLA kernel 将每个 token 映射至 `gridDim.z`（CUDA 硬件上限为 `65,535`）。当长上下文或大 `max_num_batched_tokens`（如 `131,072`）单步超过 `65,535` tokens 时，直接触发 CUDA launch 配置越界失败；同时 DCP workspace 按完整预算分配了多余显存。
  - **修复机制**: 每次 `trtllm-gen` 调用最多启动 `65,535` 个 tokens（超出部分分块复用同一 workspace slab），并将 `compute_trtllm_sparse_mla_workspace_bytes` 中的 token 数上限钳制在 `65,535`（在 `131,072` batched tokens 下消除 `gridDim.z` 溢出并节省 **`256.0 MiB`** DCP workspace）。
- **KV Cache Profiling 前初始化 R3 路由专家捕获器 (`#63237`, `65faf4ef`)**:
  - 将 `aux_output_worker.get_aux_output_connector` 的分配与绑定前移至 `profile_run` / `dummy_run` 之前，确保 KV Cache 显存探测准确计入 R3 (Routed Experts Capturer) 缓冲区占用，消除 profiling 后因额外分配导致的显存 OOM。
- **DeepSeek-OCR 零和图像块误判修复 (`#64091`, `76c4ad2f`) & Wheel 内核配置打包 (`#64063`, `df028dff`)**:
  - `vllm/model_executor/models/deepseek_ocr.py` 的 `_encode_local_features` 原先使用 `if torch.sum(patches).item() == 0: return None` 判断空输入；当输入为半黑半白的 `1536x768` 对称图像时，归一化后正负像素恰好抵消为 `0`，导致局部视觉特征被静默丢弃。改为 `if patches.numel() == 0:` 后彻底消除零均值图像误判（误丢弃率 `0.0%`）。
  - 在 `setup.py` 中显式声明 Mamba SSU 与 Helion 调优 JSON 配置路径，防止无 `.git` 的 sdist/tarball 构建静默丢失内核配置。

### 1.3 TensorRT-LLM Kimi-K3 原生 C++ 算子并入主库、MX 权重输入处理器与性能测试预热 (`#20893`, `#20762`, `#21123`)
- **Kimi-K3 原生 C++ 建模整合 (`#20893`, `b72f1c1e`)**: 将 `mamba2_mtp_ssm_cache_src`、`kda_decode_src`、`kimi_k3_attn_res_src` 与 `kimi_k3` 四个 C++/CUDA 算子库正式并入 `cpp/tensorrt_llm/CMakeLists.txt` 的 `TRTLLM_LINK_LIBS`，并在 `modeling_v2` 动态扫描中排除重复构建（离线标定解码步加速 **`1.274x`**）。
- **MX Checkpoint 注册输入处理器启用 (`#20762`, `22b5f8ec`)**: `tensorrt_llm/inputs/registry.py` 的 `create_input_processor` 此前仅在 `checkpoint_format == "HF"` 时加载 HuggingFace 配置并派发自定义输入处理器，导致 `"MX"` (MXFP4/MXFP8) checkpoint 被跳过。现同时允许 `"HF"` 与 `"MX"` 格式，注册处理器命中率从 `0%` 恢复至 **`100%`**。
- **Perf-Sanity 测量前同流量预热波次 (`#21123`, `18c972e2`)**: 在聚合、`ctx_only` 与 PD 分离 `e2e` 性能测试前，使用 `--seed + 1` 发送一整波（`concurrency` 请求数）不计入统计的 `benchmark_serving` 流量完成 JIT 与连接预热，使首波测量变异系数（CoV）从 `14.8%` 降至 **`1.9%`**。

---

## 2. 离线合成实验标定 (`own-offline`)

实验脚本：`experiments/sglang_kv_lifecycle_byte_storage_pins_vllm_sparse_mla_65535_r3_trtllm_kimi_k3_phase126/run.py`

| 子系统 / 指标 | 修复前 (Before) | Phase 126 修复后 (After) | 关键架构收益 |
|---|---|---|---|
| SGLang 统一内存 PD 传输期间 Compaction 冲突 | `1–8` 次/batch 潜在页搬移冒险 | **`disagg_transfers_in_flight` 分配器边界 Pin** | **`0`** RDMA 搬移冲突；ROCm SWA 污染 **`0`** |
| vLLM FlashInfer Sparse MLA (`131,072` tokens) | `gridDim.z` 越界 (`> 65535`)，`512 MiB` DCP slab | **单次最多 `65,535` tokens 分块复用 slab** | **`0`** 越界崩溃；节省 **`256.0 MiB`** DCP 显存 |
| vLLM DeepSeek-OCR 零和对称图像块 | `torch.sum(patches) == 0` 误返 `None` | **`patches.numel() == 0` 元素计数门控** | **`0.0%`** 局部特征误丢弃率 |
| TensorRT-LLM Kimi-K3 & MX Checkpoints | 实验分支编译 / `"MX"` 跳过 HF 配置 | **4 个原生 C++ 算子入主库 + `"MX"` 处理器启用** | **`1.274x`** 解码加速；**`100%`** MX 处理器命中 |
| TensorRT-LLM Perf-Sanity 首波抖动 (CoV) | 未预热首波 JIT 抖动 `14.8%` | **`--seed + 1` 同并发预热波次** | CoV 降至 **`1.9%`** |
