# 推理服务系统：TensorRT-LLM Kimi-K3 CuTe DSL PDL 解码残差融合与 GPU 视觉像素归一化、vLLM 混合推测解码 SW 整除块定容、SGLang AITER 零拷贝跨步 Q 扩展与 R3 非输出 Rank 跳过及 Dynamo PoolRelay（Phase 182）

> 🌐 **Language / 语言切换**: **[中文](inference-trtllm-kimi-k3-pdl-attn-res-gpu-vision-norm-vllm-sw-divisor-sglang-strided-q-r3-skip-dynamo-relay-phase182.md)** | **[English](inference-trtllm-kimi-k3-pdl-attn-res-gpu-vision-norm-vllm-sw-divisor-sglang-strided-q-r3-skip-dynamo-relay-phase182.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 182（`priority: 191`）审计了 `NVIDIA/TensorRT-LLM`、`vllm-project/vllm`、`sgl-project/sglang` 与 `ai-dynamo/dynamo` 推理服务栈的 **11 个上游主干提交**：

1. **TensorRT-LLM Kimi-K3 CuTe DSL PDL 解码注意力残差融合与 GPU 端 `uint8` 视觉像素 LUT 归一化**：
   - `c8d72931196a` (#21399) 将 Kimi-K3 解码注意力残差混合（+ RMSNorm）升级为面向 `SM100/103/107`、`M=1..64` tokens（`KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS=64`）的 CuTe DSL PDL 内核（`trtllm::kimi_k3_attn_res_*`），将上一层 MoE 的残差更新折叠进当前混合层（`_DeferredMoEResidual`，每 MoE 层省去 2 次 BF16 加法），并在 verify-only 批次向 KDA 融合投影直接输出逐比特对齐的 `R128c4` MXFP8 侧输出（配合 `fp8_block_scale_low_m_gemm` 的 gated/sigmoid/prequantized 变体）。在 GB300（PDL 生产者之后的 CUDA Graph）上，`M=7..64` 每次调用耗时从 C++ 内核的 `3.3–11.0 us` 降至 **`1.7–2.6 us`（最高 `4.231x` 加速）**；在 Rubin（`sm_107a`）`M=32, N=9` 折叠 MoE 残差时从 `6.01 us` 降至 **`2.83 us`（`2.124x`）**；Rubin `helix16 b1 mtp7` 解码整步耗时从 `11.45 ms` 降至 **`9.93 ms`（`-13.28%`）**。
   - `0160bf7a89f4` (#20973) 新增 `multimodal_config.vision_gpu_normalize: true`（Qwen3-VL 与 Gemma 4 支持）：图像缩放后保持 `uint8` 像素传输至 GPU（H2D 视觉像素带宽减半，**`2.0x` 压缩**），并在视觉编码器前通过每通道 256 项查找表（LUT）在 GPU 上完成逐比特一致的归一化。在 4x GB200 运行 `Qwen3-VL-235B-A22B NVFP4`（`c=8`）时，HTTP CPU 预处理耗时下降 **`11.2%–19.27%`**（冷启动对话从 `110.0 ms` 降至 `88.8 ms`），稳态截图会话平均 TTFT 从 `183.2 ms` 降至 **`151.8 ms`（`-17.14%`）**，稳态多模态对话 TTFT 从 `93.5 ms` 降至 **`80.4 ms`（`-14.01%`）**。
2. **vLLM 混合推测解码滑动窗口（SW）整除块定容、ModelOpt CUTLASS FP8 IPC 重启与惰性 KV 卸载队首回收**：
   - `59519ec0ee0b` (#60252) 在 `Attention.get_kv_cache_spec` 的 `_largest_kernel_block_within` 中引入 `divisor_of` 约束，确保当 GDN 混合目标模型（主块 `1,648` tokens）搭配受限于 `[16, 32, 64]` 块的 SW DFlash2 草稿模型时，SW 块保持为主块的约数并由 `unify_kv_cache_spec_page_size` 整数倍缩放，避免调度器 LCM 块从 `1,648` 膨胀 4 倍至 `6,592` tokens 及细粒度前缀缓存被禁用。在单卡 RTX PRO 5000（`sm_120`，`Qwen3.8-27B NVFP4 + DFlash2 k=7`）上，前缀缓存命中率从 `43.8%` 跃升至 **`71.2%`（`+27.4` 个百分点）**；在 DGX Spark（`GB10`）168 轮智能体回放中，总耗时从 `26.1 min` 降至 **`20.0 min`（`1.305x` 加速，命中率 `56.3% -> 77.4%`）**。
   - `69d27bf8002a` (#64119) 支持序列化 ModelOpt `FP8_PB_WO` 线性权重在 CUTLASS 后端下的跨引擎重启 CUDA IPC 权重缓存复用（B300 `116/116` 层回归通过）；`a92efadb30f5` (#59383) 在 `BlockPool.free_blocks` 中新增 `evict_first=True`，使 `SimpleCPUOffloadScheduler` 惰性卸载完成的 GPU 块归还至空闲队列的淘汰端（队首）而非 MRU 端，消除 CPU 缓存层小于 GPU KV 缓存时的无效循环刷写。
3. **SGLang AITER 零拷贝跨步 Q 统一扩展、R3 非输出 Rank 专家捕获跳过、DSpark Mamba 追踪去重与请求退役器统一**：
   - `27aedf7f93cc` (#47822) 在 AMD AITER `_forward_extend_unified` 中新增 `_unified_extend_q` 步幅守卫：当融合 RoPE + KV 写入后的 Q 列切片在每个 token 内部已经是稠密的（`stride(2)==1, stride(1)==head_dim`）时，直接返回零拷贝 `view(-1, H, D)` 而非无条件执行 `q.contiguous()`（在 `gpt-oss-120b` 36 层上每步省去 `2.0–4.3 ms` 拷贝）。在 `gfx1250`（7,548 tokens）CUDA Graph 评测中，`TP2 window` 注意力耗时从 `72.7 us` 降至 **`44.2 us`（`1.645x` 加速，`-39.20%`）**，`TP1 window` 从 `252.5 us` 降至 **`196.9 us`（`1.282x`，`-22.02%`）**。
   - `744af7ce724f` (#47553) 在 R3 路由专家捕获中仅在输出 Rank（`pp0·attn-tp0·attn-cp0`）分配设备缓冲区并跳过其余非输出 Rank 的无用拷贝，在 8x H200（`Qwen3-30B-A3B`）`--tp 8` 下将设备缓冲从 `8 x 12 MB`（`96 MB`）缩减 **8 倍**至 `1 x 12 MB`，`c=1` ITL 降低 `1.4%`（`3.621 -> 3.571 ms`），`DP attention + DeepEP` 下其余 6 个 Rank 各省 `32 MiB` 显存。
   - `435418253e5e` (#47494) 移除 DSpark 解码 worker 中重复的 `prepare_mamba_track_for_verify` 调用（每步仅在 `prepare_for_verify` 刷新 `1` 次，B200/GB300 Nemotron-3.5 NVFP4 GSM8K 准确率 **`96.0%`** `192/200`）；`fd19cfb660f4` (#48327) 将步内结束请求的 **7** 条手工释放路径统一收敛至 `RequestRetirer.release_finished`（保证先清理 HiSparse 映射再释放 KV 与 Mamba 状态）。
4. **Dynamo `PoolRelay` 实时池状态共享与分离式远程预填充取消传播**：
   - `8b4ad6a1b041` (#17716) 通过 `PoolRelay` 在中继发起（Relay-initiated）与路由器发起（Router-initiated）**2** 种连接模式间共享实时目录、就绪态、服务负载与 KV 负载视图（丢失最后一个就绪 Relay 视图时立即停止路由）；`e6b5a99a7a45` (#14775) 在双端 SGLang worker 声明 `disagg_prefill_cancel_anytime_v1` 时将客户端断连取消上下文传播至远程预填充节点（兼容对上孤儿远程预填充任务降为 **`0`**）。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+--------------------------------------------------------------------------------------------------------+
| Phase 182 Inference Serving Stack: TensorRT-LLM (Blackwell/Rubin), vLLM v1, SGLang & NVIDIA Dynamo     |
+--------------------------------------------------------------------------------------------------------+
  [TensorRT-LLM Kimi-K3 CuTe DSL PDL Mix & GPU Vision LUT (#21399, #20973)]
    |-- Kimi-K3 Decode Attn-Res (#21399)--> CuTe DSL PDL M=1..64 + Deferred MoE Add + MXFP8 KDA (4.231x GB300)
    +-- GPU Vision Pixel Norm (#20973)  --> H2D uint8 (2.0x BW) + GPU Per-Channel LUT (-17.14% TTFT, -19.27% CPU)
                                        |
                                        v
  [vLLM Hybrid Spec Decode SW Divisor, FP8 IPC Cache & Lazy KV Offload (#60252, #64119, #59383)]
    |-- SW Divisor Block Sizing (#60252)--> divisor_of=1648 Avoids 6592 LCM (+27.4 pts Hit Rate, 1.305x Spark)
    |-- ModelOpt CUTLASS FP8 (#64119)   --> Block-FP8 (FP8_PB_WO) CUDA IPC Weight-Cache Engine Restarts (B300)
    +-- Lazy KV Offload Order (#59383)  --> BlockPool.free_blocks(evict_first=True) Prevents CPU Tier Churn
                                        |
                                        v
  [SGLang AITER Strided-Q, R3 Rank Skip, DSpark Mamba & Dynamo PoolRelay (#47822, #47553, #47494, #17716)]
    |-- AITER Strided-Q Extend (#47822) --> Skip q.contiguous() on Dense Token Slices (1.645x gfx1250 TP2 Window)
    |-- R3 Capture & DSpark (#47553)    --> Output-Rank-Only 1x12MB Buf (8.0x Reduction) + 1x Mamba Verify Track
    +-- Dynamo PoolRelay & Cancel (#17716)-> Shared Live PoolViews (2 Modes) + Remote Prefill Cancel Propagation
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/trtllm_kimi_k3_pdl_attn_res_vllm_sw_divisor_sglang_strided_q_phase182/run.py`](../../experiments/trtllm_kimi_k3_pdl_attn_res_vllm_sw_divisor_sglang_strided_q_phase182/run.py) `--check` 100% 验证通过：
- `trtllm_kimi_k3_gb300_attn_res_max_speedup = 4.231`、`trtllm_kimi_k3_rubin_folded_moe_speedup = 2.124`、`trtllm_kimi_k3_rubin_mtp7_step_reduction_pct = 13.28`
- `trtllm_vision_gpu_norm_ttft_reduction_pct = 17.14`、`trtllm_vision_http_cpu_reduction_pct = 19.27`、`trtllm_vision_h2d_bandwidth_compression_ratio = 2.0`
- `vllm_sw_lcm_inflation_reduction_ratio = 4.0`（`6592 -> 1648`）、`vllm_rtx_pro_5000_prefix_hit_gain_pts = 27.4`、`vllm_dgx_spark_agentic_replay_speedup = 1.305`
- `sglang_aiter_strided_q_tp2_window_speedup = 1.645`、`sglang_r3_tp8_buffer_reduction_ratio = 8.0`、`sglang_retirer_consolidated_finish_callsites = 7`
- `dynamo_pool_relay_connection_modes = 2`、`dynamo_orphaned_remote_prefill_on_compatible_cancel = 0`

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **Blackwell / Rubin 混合注意力解码与多模态 VLM 预处理** | 在 GB300/Rubin Kimi-K3 解码（`M <= 64`）中启用 CuTe DSL PDL 注意力残差融合与延迟 MoE 残差折叠（`#21399`，`4.231x`），并在 Qwen3-VL / Gemma 4 多模态服务中启用 `vision_gpu_normalize: true`（`#20973`，TTFT `-17.14%`）。 | 对缩放前需在 FP32 中插值的视觉处理器（如 Nemotron VL）强开 `vision_gpu_normalize`（应保留在 CPU 路径以维持数值一致性）。 |
| **GDN + 滑动窗口混合推测解码与 CPU 二级 KV 卸载** | 在 vLLM 运行 Qwen3.8 + DFlash2 等 GDN/SW 混合模型时启用 `divisor_of` 块对齐（`#60252`，前缀命中率 `+27.4 pts`），并在 CPU 卸载池小于 GPU KV 池的惰性卸载场景启用 `evict_first`（`#59383`）。 | 在未打 `#55601` Mamba 状态列种子修复补丁的旧版上仅靠块缩放掩盖状态越界问题。 |
| **AMD `gfx1250` 长上下文预填充、MoE 路由采集与分离式路由** | 在 SGLang + AITER `gfx1250` 上启用 `_unified_extend_q` 零拷贝视图（`#47822`，TP2 窗口注意力 `1.645x`），在 R3 采集中启用非输出 Rank 跳过（`#47553`），并在 Dynamo 分离式部署中开启 `PoolRelay`（`#17716`）。 | 在未合入 SGLang `#47250` 异步 KV 读取完成保护的旧版 0.5.23 镜像上盲目在解码派发后立即回收远端预填充源 KV（`#14775` 安全边界提示）。 |
