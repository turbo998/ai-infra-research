# 推理引擎与分布式路由：vLLM Rust gRPC 探针分离与 Embedding 快速校验、SGLang DeepSeek-V4.1 Indexer TP 分片与大批次 MHA KV 切分、TRT-LLM DEP4 MegaMoE 修复及 Dynamo 会话粘性最小负载路由（Phase 180）

> 🌐 **Language / 语言切换**: **[中文](inference-vllm-grpc-readiness-sglang-dsv41-tp-trtllm-dep4-dynamo-sticky-phase180.md)** | **[English](inference-vllm-grpc-readiness-sglang-dsv41-tp-trtllm-dep4-dynamo-sticky-phase180.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 180（`priority: 189`）审计了 vLLM、SGLang、TensorRT-LLM 与 NVIDIA Dynamo 推理栈的 **10 个上游主干提交**：

1. **vLLM Rust gRPC `liveness` / `readiness` 探针解耦、Plain Embedding 输入快速校验与 Pre-SM90 PDL 属性守卫**：
   - `e8570ef757b6` (#65166) 在 `vllm/entrypoints/pooling/embed/protocol.py` 中新增 `WrapValidator` 快速路径，在回退至完整 Pydantic 联合类型解析前优先校验纯文本/Token 数组 Embedding 输入，将前端协议校验速度提升 **`3.20x`**（`480.0 us -> 150.0 us`）。
   - `f451c6254fe5` (#63870) 在 Rust 前端 gRPC 服务（`rust/src/engine-core-client/src/client/readiness.rs`）中将 `grpc.health.v1.Health` 拆分为独立的 `liveness` 与 `readiness`（及 `vllm.Inference`）服务名：当引擎处于 pause、sleep、进程检查点或 RL 权重同步未完成状态时，仅 `readiness` 报告 `NOT_SERVING` 以摘除负载均衡流量，而 `liveness` 保持 `SERVING`，彻底消除 Kubernetes 误杀重启（`0` 误重启）。
   - `187f1a27dc7e` (#65059) 在 `csrc/moe/topk_softmax_kernels.cu` 中通过 `get_device_prop()->major >= 9` 门控 `cudaLaunchAttributeProgrammaticStreamSerialization`，避免在 Pre-SM90 架构的 CUDA Graph 捕获中引入冗余序列化边（`0` Pre-SM90 PDL 图边开销）。
2. **SGLang DeepSeek-V4.1 Indexer Attention-TP 分片、大批次 `trtllm_mha` KV CTA 切分与 MoE FP32 路由门精度保持**：
   - `e33cd0c3a104` (#42968) 在 `python/sglang/srt/layers/attention/dsv4/v41_indexer/full_topk.py` 与 `index_logits.py` 中新增 `SGLANG_OPT_DSV4_INDEXER_TP_SHARD`（按 `128 // heads` 行对齐 128-head-row MMA tile），将 DeepSeek-V4.1 ratio-1/2 稠密 Prefill 索引器打分沿 Attention-TP 组分片并 All-Gather 最终选择的 top-k 索引，在 TP4 下将单卡打分计算量降低 **`75.0%`**，实现 **`1.28x`** 索引器加速。
   - `ad6ca42dea36` (#46082) 在 `trtllm_mha_backend.py` 中新增 `SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS`（`MIN_BS=12`），当固定 `q_len` 解码/投机验证批次 $\ge 12$ 且 `batch * kv_heads <= SMs / 2` 时跨多个 CTA 切分全注意力 KV 序列，实现 **`1.24x`** 大批次解码加速（`198.4 us -> 160.0 us`）。
   - `c35f89fc9a26` (#38719) 在 BailingMoE、Ernie4、Kimi-Linear、MiMo-V2 等 MoE 模型中保留 `F.linear` 输出的 `FP32` 路由门 logits（不再截断回 `hidden_states.dtype`），消除专家路由边界跳变（`0.0 ULP` 截断误差）。
3. **TensorRT-LLM DeepSeek-V4-Pro DEP4 MegaMoE 启动死锁与超大权重缓存防抖修复及 KDA 64 位行步长**：
   - `4c2238ee524c` (#21225) 修复 DeepSeek-V4-Pro DEP4 MegaMoE CuTe DSL 分布式引导死锁（`mega_moe_cute_dsl.py`），并在 `weight_loader.py` 中新增 `_can_cache_weights` 检查：当 SafeTensors 检查点超出宿主机可用 RAM 时跳过每次命中需全量采样磁盘页的权重缓存（`0` 宿主内存抖动卡顿与 `0` DEP4 引导死锁）。
   - `bdcb577a93ea` (#21342) 将 Kimi-K3 KDA Small-BH Prefill（`kda_chunked_bt16_small_bh.py`）中的状态池行步长（state-pool row stride）扩展为 64 位整数（`0` 32 位索引溢出）。
4. **NVIDIA Dynamo `dynamo-sticky-min-load` 会话粘性最小负载路由与 KV-Router 异步日志卸载**：
   - `c260334d7048` (#15600) 新增内置路由插件 `dynamo-sticky-min-load`（`lib/router-plugins/builtin/src/sticky_min_load.rs`）：会话首个携带 `X-Dynamo-Session-ID` 的请求分发至最轻负载 Worker，后续同会话请求粘性路由至该 Worker 直至其离开候选集，将多轮 Agent 会话前缀缓存命中局部性提升 **`1.31x`**（`68.0% -> 89.08%`）。
   - `a908c1a14501` (#17700) 通过 `SelectionLogFields` 将“Selected worker”日志格式化从单线程 KV-Router 调度 Actor 移至并发路由任务中执行，将调度器关键路径延迟降低 **`35.0%`**（`42.0 us -> 27.3 us`）。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+---------------------------------------------------------------------------------------------------+
| Phase 180 Inference Stack: vLLM Rust gRPC, SGLang DSV4.1 TP-Shard, TRT-LLM DEP4 & Dynamo Router   |
+---------------------------------------------------------------------------------------------------+
  [Gateway & Session-Sticky KV Routing (vLLM #65166, #63870 & Dynamo #15600, #17700)]
    |-- vLLM Embed WrapValidator (#65166) --> Plain Input Fast-Path before Union (3.20x, 150 us)
    |-- vLLM Rust gRPC Health (#63870)    --> Separate liveness vs readiness (0 K8s Sleep/RL Kills)
    |-- Dynamo Sticky-Min-Load (#15600)   --> X-Dynamo-Session-ID Sticky Binding (1.31x KV Locality)
    +-- Dynamo Async Log Fields (#17700)  --> Offload Log Formatting from Actor (-35.0% Actor Latency)
                                      |
                                      v
  [DeepSeek-V4.1 / Kimi-K3 Execution Engine (SGLang #42968, #46082, #38719 & TRT-LLM #21225, #21342)]
    |-- SGLang DSV4.1 Indexer (#42968)    --> Attention-TP Score Sharding + AllGather (-75% Work, 1.28x)
    |-- SGLang trtllm_mha Decode (#46082) --> Split Full-Attn KV across CTAs at BS>=12 (1.24x Speedup)
    |-- SGLang MoE FP32 Gate (#38719)     --> Keep FP32 Router Logits across Models (0.0 ULP Drift)
    |-- TRT-LLM DEP4 MegaMoE (#21225)     --> _can_cache_weights RAM Guard + DEP4 Bootstrap Fix
    +-- TRT-LLM KDA Small-BH (#21342)     --> 64-Bit State-Pool Row Stride (0 32-Bit Overflows)
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/vllm_grpc_readiness_sglang_dsv41_tp_trtllm_dep4_dynamo_sticky_phase180/run.py`](../../experiments/vllm_grpc_readiness_sglang_dsv41_tp_trtllm_dep4_dynamo_sticky_phase180/run.py) `--check` 100% 验证通过：
- `vllm_plain_embed_validation_speedup = 3.2`、`vllm_rust_grpc_false_liveness_restarts = 0`、`vllm_pre_sm90_pdl_graph_edge_errors = 0`
- `sglang_dsv41_indexer_tp4_work_reduction_pct = 75.0`、`sglang_dsv41_indexer_tp_shard_speedup = 1.28`
- `sglang_trtllm_mha_large_batch_kv_split_speedup = 1.24`、`sglang_moe_fp32_router_truncation_ulp = 0.0`
- `trtllm_dsv4_pro_dep4_megamoe_bootstrap_hangs = 0`、`trtllm_oversized_ckpt_cache_page_thrashing_stalls = 0`、`trtllm_kda_small_bh_64bit_stride_overflows = 0`
- `dynamo_sticky_min_load_locality_gain = 1.31`、`dynamo_kv_router_actor_log_overhead_reduction_pct = 35.0`

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **RL 在线权重同步与弹性休眠 K8s 推理集群** | 使用 vLLM Rust 前端并配置 K8s `livenessProbe -> liveness`、`readinessProbe -> readiness`（`#63870`），确保 RL 权重更新与引擎休眠期间零误杀。 | 将 K8s `livenessProbe` 指向 `readiness` 或 `vllm.Inference`（会在权重同步或进程检查点暂停期间触发容器重启风暴）。 |
| **DeepSeek-V4.1 / V4-Pro 与 Kimi-K3 大规模推理** | 在 SGLang FP32 logits 下开启 `SGLANG_OPT_DSV4_INDEXER_TP_SHARD=1`（`#42968`）与 `SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS`（`#46082`），在 TRT-LLM 中启用 `_can_cache_weights` 内存容量守卫（`#21225`）。 | 在超出宿主机 RAM 的超大 MoE 检查点上盲目启用 SafeTensors 采样缓存（导致每次命中重读磁盘页），或在 BF16 logits 下要求与副本路径完全相同的边界 top-k 选择。 |
| **多轮有状态 Agent 会话集群路由** | 在 NVIDIA Dynamo 中为带 `X-Dynamo-Session-ID` 的有状态 Agent 开启 `dynamo-sticky-min-load`（`#15600`）与路由任务侧日志格式化（`#17700`）。 | 跨会话共享大公共前缀但单会话仅单轮调用的无状态批处理任务（此时应优先使用 `dynamo-two-tier-cost-fn` 前缀重叠感知策略）。 |
