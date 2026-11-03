# 推理与分离式服务栈：vLLM DeepSeek V4/V4.1 打包视觉编码与在线 FP8/NVFP4 权重缓存重启、SGLang 单 Warp MoE 路由、TRT-LLM 8-Token 分块 QKV 预处理与 GB300 Host Cache 归零及 Dynamo 立即解码终端转发（Phase 181）

> 🌐 **Language / 语言切换**: **[中文](inference-vllm-dsv4-packed-vision-weight-cache-sglang-warp-moe-router-trtllm-qkv-tiling-gb300-host-cache-dynamo-handoff-phase181.md)** | **[English](inference-vllm-dsv4-packed-vision-weight-cache-sglang-warp-moe-router-trtllm-qkv-tiling-gb300-host-cache-dynamo-handoff-phase181.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 181（`priority: 190`）审计了 vLLM、SGLang、TensorRT-LLM 与 NVIDIA Dynamo 的 **11 个上游主干提交**：

1. **vLLM DeepSeek V4/V4.1 Vision `forward_packed` 多图打包编码、Fast-Start 在线 FP8 / CUTLASS NVFP4 权重缓存 IPC 重启与 ShortConv (`LFM2`) 批次不变性**：
   - `0eeae6a47333` (#65030) 将 DeepSeek V4 与 V4.1 Vision 单次请求内的多张图像通过 `forward_packed` 合并为单次 ViT + Aligner 打包前向（复用 V4.1 编码器 CUDA Graph 路径），在 H100 54 个多图调用基准上将编码速度提升 **`2.61x`**（V4，32 图从 `962.9 ms` 降至 `328.4 ms`）与 **`2.10x`**（V4.1，`961.2 ms -> 418.6 ms`）；在 2x B200 真实 `DeepSeek-V4-Flash-Vision-Exp` 上将 4 图并发 1 的中位 TTFT 从 `367–378 ms` 降至 **`271 ms`**，并将 VisionArena 并发 16 的中位 TTFT 从 `914–959 ms` 降至 **`548 ms`**（**`1.709x`** 加速，MMMU-Pro 1730 题准确率 `862 vs 861/860` 保持一致）。
   - `44965a10974e` (#63672) 支持权重缓存守护进程（weight-cache daemon）跨引擎 IPC 重启直接复用在线 FP8/MXFP8 线性层权重与 ModelOpt NVFP4 CUTLASS MoE 权重，无需二次量化或转换，修复了 IPC 跳过后处理时未填充 CUTLASS FP8 输出宽度丢失以及 NVFP4 激活缩放因子被二次乘入共享权重缩放因子的缺陷（`0` 二次缩放错误）。
   - `9226d6e59cd1` (#62367) 为 `SHORT_CONV_ATTN`（`Lfm2ForCausalLM`、`Lfm2MoeForCausalLM`）启用 `VLLM_BATCH_INVARIANT=1` 批次不变模式，`LiquidAI/LFM2-8B-A1B` 100% 通过 `test_batch_invariance.py`（`22/22`），消除开启前缀缓存时第 0 轮与后续轮次的贪婪解码漂移。
2. **SGLang 单 Warp/Token CUDA MoE 路由器 (`moe_route_warp`)、`HashTopK` 预填充图守卫、Ascend 950 ModelSlim MLA FP8 KV Cache 与语法超时防毒化**：
   - `4494d682373e` (#47641) 新增 JIT CUDA MoE 路由器 `moe_route_warp`（每 Token 分配 1 个 Warp，每个线程通道将 $E/32$ 个专家保存为保序 `u32` 键并通过编译期 Batcher 排序网络一次性排序，每轮 top-$k$ 仅对 32 个通道头执行 `redux` max 与最小通道 tie-break，覆盖 $128\dots 512$ 专家的 sigmoid / sqrtsoftplus / softmax 路由），在 B200（CUDA Graph）上将 Qwen3.5（512 专家，top-10）`T=16384` 路由延迟从 `63.1 us` 降至 **`15.6 us`**（**`4.045x`** 加速，`T=1` 从 `3.40 us` 降至 `1.68 us`），Kimi-K2（384 专家）从 `39.7 us` 降至 **`13.1 us`**（**`3.031x`**），DeepSeek-V3（256 专家）从 `24.6 us` 降至 **`12.3 us`**（**`2.000x`**）。
   - `1ef3335da114` (#48041) 在可中断预填充（breakable-prefill）双流 CUDA Graph 捕获与视觉 top-$k$ 分支中优先检查 `is_hash` 再读取 `topk_config`，消除 `HashTopK` 层抛出 `AttributeError`；`99fd01047d67` (#46921) 在华为 Ascend 950 NPU 上支持 DeepSeek-V3 / Kimi-K3 的 ModelSlim 静态 `Q_FP8_DYNAMIC_KV_FP8` 潜在 KV 缓存量化与 `MLAProlog`；`86ab345175d0` (#45466) 修复 `GrammarManager` 在超过 `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` 时将排队超时错误缓存为 `InvalidGrammarObject` 的缺陷，消除突发负载下 JSON Schema 被永久毒化拒绝的问题。
3. **TensorRT-LLM 8-Token 分块 QKV 预处理、GB300 Nemotron-Ultra-V3 Host Cache 归零消除 NUMA 碎片整理抖动与分离式 KV 传输超时归因**：
   - `a6a8e8d67183` (#21198) 新增 `TRTLLM_ENABLE_QKV_ROPE_TILING=1` 8-token 分块融合 QK-norm + RoPE + gate 提取路径（当 `tokens >= 256` 时生效），在 B200 上将 `4,096` 与 `32,737` tokens 的预处理速度分别提升 **`2.41x`** 与 **`2.67x`**。
   - `508f8dddefa8` (#21550) 定位并修复了 4x GB300 上 Nemotron-Ultra-V3 FP4（`TEP4 MTP6`）的性能回退：默认未设置 `host_cache_size` 时，KV Cache Manager V2 在启动末尾自动分配并锁页（pin）每 Rank **`86.24 GiB`**（目标模型 `79.61 GiB` + 草稿模型 `6.63 GiB`）Host 层内存，导致 27 次使 NUMA 节点在 512 MiB 大页阶数上碎片率 $>90\%$ 的分配全部触发持续 `13–49 s` 的 Linux 内核主动内存规整（proactive compaction），使解码步中位延迟从 `8.10–8.47 ms` 恶化至 `8.88–9.02 ms`（吞吐跌至 `326.28–382.78 tok/s`）。显式设置 `host_cache_size: 0` 后，中位吞吐达 **`390.54 tok/s`**，低于 `385 tok/s` 的劣化运行从 `10/24` 降至 `1/24`，低于 `367.24 tok/s`（基线 -5%）的运行降至 **`0/24`**。
   - `da980cce4754` (#20708) 将分离式 KV 传输超时精确归因至截止时间持有者（`executor-request`、`native-sender`、`retirement-grace`）与非阻塞 TX/RX 已观测字节进度；`c26c166b8e49` (#21199) 在请求准入阶段拒绝超过单次编码器迭代上限（`max_num_tokens` / `encoder_max_num_tokens` / 交叉注意力 `max_seq_len`）的 Encoder-Decoder 请求。
4. **NVIDIA Dynamo 分离式 Prefill-Router 解码终端立即转发与无 GPU Mocker 池声明式成本仲裁**：
   - `2ec0b1112d00` (#17484) 修复了 SGLang 引导交接（bootstrap handoff）分离式服务中的尾部阻塞：此前 `PrefillTask::forward_decode` 会扣留已成功生成的解码终端块（含最后生成的 tokens 与 `finish_reason`），直至晚约 `1.5` 个预填充调度周期的 Prefill 响应返回（占最慢 10% 请求的 94%）。修复后一旦解码成功即刻向客户端转发解码终端（仅在流关闭阶段等待 Prefill 结束并忽略迟到的 Prefill 失败），将多轮 Agent 基准测试的每用户输出吞吐 P90 从 **`360 tok/s` 提升至 `470 tok/s`（`1.306x` / `+30.6%`）**；`c80a17b4631e` (#14055) 支持在 Global Planner 中为 DGD 请求 0 GPU 的 Mocker 池声明虚拟 GPU 成本以参与预算仲裁。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+------------------------------------------------------------------------------------------------------+
| Phase 181 Inference & Disaggregated Serving: vLLM, SGLang, TensorRT-LLM & NVIDIA Dynamo              |
+------------------------------------------------------------------------------------------------------+
  [vLLM Multimodal, Fast-Start Weight-Cache IPC & Determinism (#65030, #63672, #62367)]
    |-- DeepSeek V4/V4.1 Vision (#65030)--> forward_packed Multi-Image Pass (2.61x V4, c=16 TTFT 548 ms)
    |-- Fast-Start Weight Cache (#63672)--> Online FP8/MXFP8 & ModelOpt NVFP4 MoE IPC Restart (0 Re-Scale)
    +-- LFM2 ShortConv (#62367)         --> VLLM_BATCH_INVARIANT=1 Support (22/22 Determinism Tests Pass)
                                      |
                                      v
  [SGLang Warp MoE Router, TRT-LLM B200/GB300 & Dynamo Disagg Handoff (#47641, #21198, #21550, #17484)]
    |-- SGLang moe_route_warp (#47641)  --> 1-Warp/Token Batcher Sort + redux Top-K (4.045x Qwen3.5 512E)
    |-- SGLang Reliability (#48041/#45466)-> HashTopK Dual-Stream Guard + No Grammar Poll-Timeout Poisoning
    |-- TRT-LLM B200/GB300 (#21198/#21550)-> 8-Tok QKV Tiling (2.67x) + GB300 host_cache_size:0 (-86.24 GiB)
    +-- Dynamo Prefill-Router (#17484)  --> Immediate Decode Terminal Forward (P90 360 -> 470 tok/s, 1.306x)
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/vllm_dsv4_packed_vision_sglang_warp_moe_trtllm_qkv_tiling_gb300_host_cache_dynamo_phase181/run.py`](../../experiments/vllm_dsv4_packed_vision_sglang_warp_moe_trtllm_qkv_tiling_gb300_host_cache_dynamo_phase181/run.py) `--check` 100% 验证通过：
- `vllm_dsv4_packed_vision_54call_speedup = 2.61`、`vllm_dsv41_packed_vision_54call_speedup = 2.10`、`vllm_dsv4_b200_c16_ttft_speedup = 1.709`（`548.0 ms`）
- `vllm_weight_cache_nvfp4_double_scale_bugs = 0`、`vllm_lfm2_shortconv_batch_invariance_pass_rate = 1.0`
- `sglang_moe_route_warp_qwen35_t16384_speedup = 4.045`、`sglang_moe_route_warp_kimik2_t16384_speedup = 3.031`、`sglang_moe_route_warp_dsv3_t16384_speedup = 2.0`
- `trtllm_b200_qkv_rope_tiling_4k_speedup = 2.41`、`trtllm_b200_qkv_rope_tiling_32k_speedup = 2.67`
- `trtllm_gb300_nemotron_host_pin_avoided_gib_per_rank = 86.24`、`trtllm_gb300_nemotron_median_tok_s = 390.54`、`trtllm_gb300_nemotron_runs_below_367_tok_s = 0`
- `dynamo_disagg_decode_terminal_p90_speedup = 1.306`（`360 -> 470 tok/s`）

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **多图视觉大模型（VLM）与高专家数 MoE 在线服务** | 在 vLLM 上部署 DeepSeek V4/V4.1 Vision 时默认启用 `forward_packed` 多图打包编码（`#65030`，`2.61x` 编码加速、`c=16` TTFT `548 ms`），并在 SGLang B200/H200 上对 $128\dots 512$ 专家模型启用 `moe_route_warp`（`#47641`，Qwen3.5 `4.045x`）。 | 32 图极端并发且显存余量 `< 2 GiB` 的场景需注意 `forward_packed` 编码器峰值显存增加 `1.7–2.0 GB`；`>512` 专家（如 Kimi-K3 896 专家）仍走 Triton 路由器。 |
| **Grace Blackwell (GB300) 大模型服务与快速热重启** | 在 GB300 上当单请求/低 KV 缓存占用工作负载不需要 Host KV 二级卸载时显式配置 `host_cache_size: 0`（`#21550`，避免每 Rank 锁页 `86.24 GiB` 触发 Linux 512 MiB 大页主动规整抖动），并在 vLLM 热重启中启用在线 FP8 / NVFP4 权重缓存 IPC 复用（`#63672`）。 | 在需要严格跨批次逐 token 比特一致性的场景中盲目开启 `TRTLLM_ENABLE_QKV_ROPE_TILING=1`（`#21198` 分块会改变 Q/K 舍入边界）。 |
| **分离式（Disaggregated）多轮 Agent 推理** | 在 Dynamo + SGLang 引导交接分离式集群中启用解码终端立即转发（`#17484`，P90 用户吞吐 `+30.6%`）与 TRT-LLM KV 传输超时持有者进度归因（`#20708`）。 | 在解码流尚未读取到 EOF 前提前主动断开底层 worker 流（会导致 worker 报 `Failed to publish complete final`）。 |
