# SGLang ComfyUI SGLDiffusion、StaticKVStorage 与路由 Draft KV 视图、vLLM P2P Retained Source Pins、SWA 惰性卸载与 DSA Decode Indexer 显存预留、NIXL Push、ROCm MXFP8 融合及 TensorRT-LLM 64 位 AttnRes 步长深挖（Phase 124）

> **English Edition**: [sglang-comfyui-kv-storage-routed-views-vllm-p2p-retain-swa-dsa-nixl-rocm-trtllm-phase124.en.md](sglang-comfyui-kv-storage-routed-views-vllm-p2p-retain-swa-dsa-nixl-rocm-trtllm-phase124.en.md)
> **证据等级**：`implementation` + `author-benchmark` + `own-offline`（零 GPU 硬件重测声明）

---

## 1. 背景与核心架构问题

在多模态扩散工作流、PD 分离（Disaggregated Prefill/Decode）、P2P RDMA KV 卸载以及超长上下文稀疏注意力（DSA / Kimi-K3 AttnRes / GLM-5.3-Flash KPool）推理引擎中，Phase 124 Track B 解决了十一个生产级可靠性与性能缺陷：

1. **ComfyUI SGLDiffusion 适配器静默丢弃不支持的引导参数 (`4517a5ff1b`, `b1c6945659`)**：ComfyUI LTX2 / Wan / Lumina2 节点向 SGLDiffusion worker 传递 `attention_mask`、`additional_t_cond`、`siglip_feats`、`ref_contexts` 时曾被静默忽略，产出无条件错误图像；且未开启 `--disable-cuda-malloc` 时无法启用零拷贝 CUDA IPC。
2. **SGLang KV 池字节存储与分配索引空间耦合 (`146378629d`, `53592ea5ff`, `ca8184e13a`)**：Beam-search fork、PD 分离传输与主机 KV 卸载各自绕过 `KVCache` 生命周期契约，且将字节拷贝方法与 `IdSpaceKind`（token slot、page block、MHA window slot、linear state slot）混用。
3. **SGLang 投机采样路由视图缺失、RDMA 传输期间页面压缩竞争与 ROCm 滑窗越界 (`240862923e`, `eb79ecca11`, `14fea75124`)**：融合投机 draft 未绑定 `get_routed_kv_views`；PD 分离 RDMA 传输在飞行中未在分配器边界登记 `disagg_transfers_in_flight` 导致被页面压缩器迁移；ROCm 在滑窗地址转换池上错误开启 `enable_fused_set_kv_buffer`。
4. **vLLM P2P KV 卸载断开连接时的 DMA 源缓冲区 Use-After-Free (`a1cfd2c1b5`)**：`remove_remote_peer()` 或 `session.close()` 在取消飞行中 `WRITE` 传输后立即释放源内存块 pin，而远端 RDMA NIC 仍在执行异步 DMA 读取，且对端重连时复用了旧 peer ID。
5. **vLLM SimpleKVOffload 漏算混合 SWA 层与 DSA Decode Indexer 显存漏计 (`77f58e5304`, `317081fb6d`)**：`_estimate_lazy_target_blocks` 仅遍历 `prefix_cacheable_groups`，漏掉了 `SlidingWindowMLASpec` 滑窗层；DSA `sparse_attn_indexer` 在显存 profiling 阶段未分配 decode indexer logits（在 `256×65536` FP32 下每次漏计 **`64.0 MiB`** 激活缓冲）。
6. **vLLM NIXL Push 模式、ROCm MiniMax-M3.1 MXFP8 融合与自适应投机 Padding (`066d979934`, `1cb31be702`, `5118d990fb`)**：`NixlConnector` 需根据 `transfer_mode: "push"` 自动分发至 `NixlPushConnector`；ROCm MiniMax-M3.1 需将 MXFP8 E4M3 量化融合进 Gemma `input_layernorm`。
7. **TensorRT-LLM Kimi-K3 `attn_res` 32 位有符号整数溢出、Eager GDN `torch.cat` 拷贝与 GLM-5.3-Flash KPool (`e2b1544160`, `99466ea9ad`, `2fe91cdfc9`)**：Kimi-K3 持久化融合 `attn_res` 核函数中 `int block_stride_r = num_tokens * hidden_dim` 在 `H = 7168, N = 9`（`max source = 7`）下，当 **`num_tokens >= 42,800`** 时计算 `7 * 42800 * 7168 = 2,147,532,800 > INT_MAX (2,147,483,647)`，发生 32 位有符号溢出回绕为负数 `-2,147,434,496` 导致非法内存访问崩溃。

---

## 2. 核心实现路径与定量证据分析

```
+---------------------------------------------------------------------------------------------------+
| Phase 124 Track B: SGLang / vLLM / TensorRT-LLM KV Lifecycle, DMA Pinning & 64-Bit Addressing     |
+---------------------------------------------------------------------------------------------------+
| 1. SGLang ComfyUI SGLDiffusion (`4517a5ff1b`, `b1c6945659`) & StaticKVStorage (`146378629d`..`14fea`)|
|    - Reject unsupported conditioning kwargs fail-fast; CUDA IPC cuts H100 Flux+CN 13.1s -> 4.7s   |
|    - StaticKVStorage (`kv_storage_of`) + IdSpaceKind + `disagg_transfers_in_flight` compaction pin|
+---------------------------------------------------------------------------------------------------+
| 2. vLLM P2P Retained Source Pins (`a1cfd2c1b5`) & SWA/DSA Memory Accounting (`77f58e5304`, `317081`)|
|    - `cancel(mode="wait")` -> `_retained` holds source block pins up to `_RETAINED_HOLD_CAP_S=30s`|
|    - `_estimate_lazy_target_blocks` sums all `kv_cache_groups` (48 full + 16 SWA -> 1.3333x blocks)|
|    - DSA `sparse_attn_indexer` reserves decode logits (`256 * 65536 * 4B = 64.0 MiB` FP32)        |
+---------------------------------------------------------------------------------------------------+
| 3. TensorRT-LLM Kimi-K3 64-Bit Stride (`e2b1544160`), Eager GDN (`99466ea9ad`) & KPool (`2fe91cdf`) |
|    - `long long block_stride_r`: eliminates 32-bit overflow at T >= 42,800 (H=7168, N=9)          |
|    - Pre-allocates shared `attn_out` `(1, T, Hv_tp, Dv)` -> zero `torch.cat` allocation/copy      |
|    - Blackwell `glm_kpool` sparse attention: +6.0% concurrency-1 output speed on 4x B300 (EP=1)   |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 SGLang ComfyUI SGLDiffusion 契约校验与 `StaticKVStorage` 架构演进
- **Fail-Fast 参数契约与 CUDA IPC 加速 (`4517a5ff1b`, `b1c6945659`)**：显式拦截并报错 `attention_mask`、`additional_t_cond`、`siglip_feats`、`ref_contexts` 等尚未透传的条件输入；新增 `SGLDWorkerStatus` 节点检验 `--disable-cuda-malloc` 零拷贝 CUDA IPC 状态。在 H100 上，Flux + ControlNet `1024×1024` 25 步推理延迟从原生 ComfyUI 的 `13.1 s` 降至 **`4.7 s`（`2.7872x` 加速）**，Wan2.2 TI2V 5B 从 `132.0 s` 降至 **`39.4 s`（`3.3503x` 加速）**。
- **`StaticKVStorage` 与 `IdSpaceKind` 解耦 (`146378629d`, `53592ea5ff`, `ca8184e13a`)**：统一使用 `kv_storage_of(pool)` 暴露 `copy_rows`、`get_contiguous_buf_infos`、`read_tokens_to_host`、`write_tokens_from_host`，并将分配标识空间严格划分为 `TOKEN_SLOT`、`PAGE_BLOCK`、`MHA_WINDOW_SLOT` 与 `LINEAR_STATE_SLOT`。
- **投机 Draft 路由视图与 PD 分离 Pin 保护 (`240862923e`, `eb79ecca11`, `14fea75124`)**：融合投机 draft 在初始化时绑定 `get_routed_kv_views`；PD 分离传输在分配器边界递增 `disagg_transfers_in_flight`，当计数 `> 0` 时阻止碎片压缩迁移飞行中页面；ROCm 上以 `not _window_writes_translated(pool)` 门控 `enable_fused_set_kv_buffer`。

### 2.2 vLLM P2P 延迟释放 Pin、SWA 惰性卸载与 DSA Indexer 显存预留
- **P2P KV 卸载 `_retained` 源内存钉死 (`a1cfd2c1b5`)**：在 `remove_remote_peer()` 与 `session.close()` 时，对飞行中的源端 `WRITE` 传输调用 `cancel(mode="wait")` 并移入 `_retained` 队列，继续持有源 KV 块引用计数（最长 `_RETAINED_HOLD_CAP_S = 30.0 s`）直至 `poll_retired()` 确认底层 DMA 停止，彻底消除 RDMA 源缓冲区 Use-After-Free 及重连 peer ID 冲突。
- **混合 SWA 惰性块估算 (`77f58e5304`) 与 DSA Decode Logits 显存预留 (`317081fb6d`)**：
  - `_estimate_lazy_target_blocks` 改为遍历全部 `kv_cache_groups`（含 `SlidingWindowMLASpec`），在 48 层全局 + 16 层滑窗混合模型上修正了 `1.3333x` 的目标块估算偏差。
  - `sparse_attn_indexer` 在 `is_Profiling` 阶段显式分配 `[num_reqs, max_seq_len]` decode indexer logits（当 `topk_backend != "deep_select" or dcp_world_size != 1` 时为 FP32），在 `256 × 65536` 下预留 **`64.0 MiB`**（BF16 为 `32.0 MiB`），消除长序列解码阶段的显存 OOM。
- **NIXL Push 与 ROCm MXFP8 融合 (`066d979934`, `1cb31be702`, `5118d990fb`)**：支持 `kv_Connector_extra_config.transfer_mode == "push"` 实例化 `NixlPushConnector`；ROCm MiniMax-M3.1 通过 `MinimaxM3RMSNormQuantFusionPass` 将块大小 32 的 MXFP8 E4M3 激活量化融合进 Gemma `input_layernorm`。

### 2.3 TensorRT-LLM Kimi-K3 64 位步长、Eager GDN 预分配与 GLM-5.3-Flash KPool
- **Kimi-K3 `attn_res` 64 位偏移修复 (`e2b1544160`)**：在 `attnResBlock` 持久化融合核函数中，源激活偏移为 $\text{offset} = \text{source} \times T \times H$。对于 Kimi-K3（$H = 7168, N = 9 \Rightarrow \max(\text{source}) = 7$），当 $T \ge 42,800$ 时：
  $$7 \times 42,800 \times 7,168 = 2,147,532,800 > 2^{31} - 1 = 2,147,483,647$$
  将 `block_stride_r` 从 32 位 `int` 提升为 64 位 `long long` 消除了长上下文 prefill 的整数溢出崩溃。
- **Eager GDN 零拷贝预分配 (`99466ea9ad`) 与 GLM-5.3-Flash KPool (`2fe91cdfc9`)**：GDN mixer 预先分配共享 `attn_out` `(1, T, num_v_heads_per_tp, head_v_dim)` 并让 prefill 与 decode 直接写入不相交切片，省去 `torch.cat` 拷贝；Blackwell `glm_kpool` 稀疏注意力优化在 4×B300（`EP=1, MTP=3, FP8 KV`）concurrency-1 下提升 **`+6.0%`** 输出吞吐。

---

## 3. 确定性实验与架构师练习

- **离线验证程序**：`experiments/sglang_comfyui_kv_storage_routed_views_vllm_p2p_retain_swa_dsa_nixl_rocm_trtllm_phase124/run.py`
- **架构师核算练习**：验证 Kimi-K3 `attn_res` 在 $H = 7168, N = 9$ 下发生 32 位有符号溢出的精确临界 token 数 $T_{\text{crit}} = \lfloor (2^{31}-1) / (7 \times 7168) \rfloor + 1 = 42,799 + 1 = 42,800$。
