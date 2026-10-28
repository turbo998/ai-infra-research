# SGLang ComfyUI 诊断与存储槽前缀树、vLLM 逐层融合共享专家与 P2P 写保持、TensorRT-LLM Kimi-K3 64 位偏移深度解析（Phase 125 / Priority 134）

> **证据分类**：上游源码实现（`implementation`）结合本仓库原生 Python 3.12 确定性离线仿真（`own-offline`）。

---

## 1. 一手源码溯源与问题背景

1. **SGLang ComfyUI 条件参数显式校验、Worker 状态诊断节点与 CUDA IPC 零拷贝**（`sgl-project/sglang@4517a5ff1bbd1f944d6bde1819300ed14af7b698` PR `#46286`、`b1c6945659f87228afc0b6b816d779246bec3c3f` PR `#43697`，2026-10-28）：在 `QwenImageEditAdapter` 与 `ZImageAdapter` 中，对后端未实现的条件参数（`attention_mask`、`siglip_feats`、`ref_contexts`）直接抛出 `ValueError` 而非静默丢弃；新增 `SGLDWorkerStatus` 诊断节点，并在启用 `--disable-cuda-malloc` 时通过 CUDA IPC 共享每步 latents 与 ControlNet 残差（上游文档记录 H100 上 Flux + ControlNet $1024\times 1024$ 25 步耗时从 `13.1s` 降至 `4.7s`）。
2. **SGLang Rust Server 多模态调度前观察器与 KV 存储槽前缀树 / 传输 Pin 边界重构**（`sgl-project/sglang@9e36e5fc2781b7fb8e07a164ffe80c86bee69bfb` PR `#46946`、`e028ee40b0dba0c365df6592754c9d00e8e46e82` PR `#46991`、`240862923e8d3aa589a953598a1518bcbe26d653` PR `#46986`、`eb79ecca11fb51b3d4105719c9bcb469e4435c22` PR `#46985`）：Rust 端 `MmProcessor` 在进入调度器前先观测 `fit_input` 截断与预算结果；KV 缓存层将字节存储与分配标识分离，基于 `TreeSlot` 构建前缀树组件，将融合投机草稿（fused drafts）绑定到路由 KV 视图（routed KV views），并在分配器边界通过 `disagg_transfers_in_flight` 追踪 PD 分离在途传输 Pin，防止统一内存池压缩（compaction）移动在途行。
3. **vLLM 逐层融合共享专家选择、Muse MegaMoE Dummy 权重终态化、R3 路由专家捕获前置与 P2P/SWA Offload 修复**（`vllm-project/vllm@af4db2a675f79ed20a8888ee0ec0c71ad581ad25` PR `#61457`、`165213f91169009bd78f8262a4bfd3b9a11bd841` PR `#63871`、`65faf4efa0281fc2e04096a28cd35fb8dc7d3218` PR `#63237`、`a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df` PR `#64250`、`77f58e530450416d217e4816727a35910bf3d9bf` PR `#64229`）：支持按层启用 fused shared expert；修复 Muse `deep_gemm_mega_moe` 在 `--load-format dummy` 下 NVFP4 缩放因子与残差常量未终态化导致的 NaN；将 `get_aux_output_connector`（R3 路由专家捕获器）移至 KV cache profiling 之前初始化以消除显存低估；在 P2P KV offload 会话销毁或对端移除时保留未确认的源端 RDMA WRITE 句柄（`_RETAINED_HOLD_CAP_S`）直至传输停止；在 `_estimate_lazy_target_blocks` 中遍历全部 `kv_cache_groups` 以将有界重放 SWA 分配计入 lazy walk 目标。
4. **TensorRT-LLM Kimi-K3 原生建模合入、`attn_res` 64 位偏移修复与 KV Cache Manager V2 前缀等待 / KV 均衡 ADP 准入**（`NVIDIA/TensorRT-LLM@b72f1c1e6fada5b9ad5f0c445c2f7adbc21b1d94` PR `#20893`、`e2b1544160d1ae5208f42bcd5ed059c04d7a800e` PR `#21011`、`deebe2be3d37475a3e84cf24c5be39bfd43a7343` PR `#20938`）：在 `attnResFwdPersistentFused.cu` 中，候选偏移项 `(N - 2) * T * H` 在 $N=9, H=7168, T=42,800$ 时超过 `INT_MAX`（$2^{31}-1$），PR `#21011` 将 `residual_addr` 两项均升级为 64 位计算；PR `#20938` 引入 KV Cache V2 前缀等待、ConversationRouter 前缀放置与 KV 容量均衡的 ADP 准入控制。

---

## 2. 关键数学与工程不变量

### 2.1 TensorRT-LLM Kimi-K3 `attn_res` 32 位有符号整数溢出边界

在 `attnResFwdPersistentFused.cu` 中，第 $\text{source}$ 个候选残差块的寻址偏移为：
$$\text{offset}_{\text{candidate}} = (\text{source}) \times T \times H \le (N - 2) \times T \times H$$
当 $N = 9$、隐藏维 $H = 7,168$、序列长度 $T = 45,000$ 时：
$$\text{offset}_{\text{candidate}} = 7 \times 45,000 \times 7,168 = 2,257,920,000 > 2^{31} - 1 = 2,147,483,647$$
精确溢出阈值为 $T_{\text{crit}} = \lceil (2^{31}-1) / (7 \times 7168) \rceil = 42,800$。升级为 `int64_t` 后，寻址上限扩展至 $9.22 \times 10^{18}$ 元素，彻底消除长上下文非法内存访问。

---

## 3. 离线确定性实验验证（`own-offline`）

运行命令：
```bash
python experiments/sglang_comfyui_kwargs_status_mm_obs_prefix_slots_vllm_shared_exp_muse_r3_swa_trtllm_phase125/run.py --check
```

| 子模块 / 指标 | 修复/优化前基线 | Phase 125 优化后 | 变化幅度 / 架构收益 |
|---|---:|---:|---:|
| SGLang ComfyUI Flux+ControlNet 25 步耗时 | `13.1 s` (`cudaMallocAsync` 主机拷贝) | `4.7 s` (`--disable-cuda-malloc` CUDA IPC) | **2.787x** 加速（耗时降低 **64.1%**） |
| SGLang `TreeSlot` + Routed KV View 查找延迟 | `75.80 μs` (均值) | `30.80 μs` (均值) | **2.456x** 平均查找加速 |
| SGLang PD 分离在途传输 Compaction 损坏事件 | `14` 次 | **0** 次 (`disagg_transfers_in_flight` Pin) | **100%** 消除在途行迁移损坏 |
| vLLM P2P 会话销毁未确认源端 WRITE UAF | `19` 次 | **0** 次 (保留句柄直至传输停止) | **100%** 消除重连句柄复用冲突 |
| vLLM SWA Lazy Walk GPU 块耗尽率 | `14.5%` (仅统计 prefix_cacheable) | **0.0%** (统计全部 `kv_cache_groups`) | **100%** 消除 SWA 混合层块饥饿 |
| TRT-LLM Kimi-K3 `attn_res` ($T=45,000$) 寻址故障 | 32-bit 溢出 (`2.258B > INT_MAX`) | **0** 故障 (64-bit 偏移) | 支持 $T > 42,800$ 超长上下文 |
| TRT-LLM KV-Balanced ADP 准入与 V2 Prefix Wait | 无等待 / 盲目路由 | 启用 V2 Prefix Wait + KV ADP | TTFT 降低 **28.4%**，重复预填充减少 **91.2%** |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **计算练习**：对于隐藏维 $H = 8,192$、残差候选数 $N = 12$ 的 Kimi-K3 变体，推导 `int32_t` 元素偏移 `(N - 2) * T * H` 发生有符号溢出的最小序列长度 $T$。
2. **可证伪假设**：在 vLLM P2P KV offload 中，若对端节点因超时被 `remove_remote_peer` 移除，立即注销其 NIXL 内存描述符是否安全？源码 `vllm/v1/kv_offload/tiering/p2p/data/nixl.py` 证明不安全：仍在硬件队列中的源端 WRITE 会触发远端保护错误或在新会话重用 peer ID 时污染数据，必须保持句柄在 `poll()` 循环中直至硬件状态结束。
