# SGLang Xeon Grok-2 与 XPU WanVideo 防护、vLLM 引用式 UVA 暂存、KV Offload P2P 分模式快速失败、CUDA 13.4 SM 局部性域 NVFP4 MoE 与 TensorRT-LLM 分离式 Responses API 解析（Phase 94）

> **English Version**: [sglang-xeon-grok2-xpu-wan-vllm-uva-p2p-loc-trtllm-phase94.en.md](sglang-xeon-grok2-xpu-wan-vllm-uva-p2p-loc-trtllm-phase94.en.md)  
> **Evidence Class**: `implementation` + `own-offline`  
> **Experiment Path**: `experiments/sglang_xeon_grok2_xpu_wan_vllm_uva_p2p_loc_trtllm_phase94/`

---

## 1. 架构背景与核心问题

Phase 94 Track 2 剖析 SGLang（`#46393`, `#46049`, `#46501`, `#43341`）、vLLM（`#63633`, `#63781`, `#61685`, `#60968`）与 TensorRT-LLM（`#19883`, `#20643`）的十项推理引擎关键演进：

1. **SGLang Xeon CPU Grok-2 原生推理与预分片 TP 偏移修复（`#43341`）**：
   - 实现了 AVX-512 向量化 FP32 累加路由点积 `router_dot_fp32_accum` 与融合 `tanh` softcapping（`30.0`）的 `fused_moe_router_cpu`；
   - 在 `NormReduceGeneric<has_post_residual=true>` 中将归一化乘权重后的激活先截断回 `BF16` 再与 `residual` 累加，严格对齐 CUDA 算子的数值轨迹；
   - 修复了 CPU 预分片 MoE 权重（`use_presharded_weights=True`）仍按 `weight_start = shard_size * moe_tp_rank` 切片导致的越界 bug（例如 `shard_size=2048, rank=3` 时错误偏移 `6144`，修复后 `weight_start = 0`）。
2. **SGLang Intel XPU WanVideo CuTe 防护、Grammar HTTP 500 状态与 HiCache 注入代理（`#46393`, `#46049`, `#46501`）**：在 Intel XPU 上跳过不支持的 CuTe DSL `fused_Add_norm_interleave_rope` 导入；当结构化输出 Grammar FSM `accept_token` 在 prefill/decode 抛异常时显式返回 `HTTPStatus.INTERNAL_SERVER_ERROR (500)` 而非伪装正常完成；并在 `StorageBackendFailureInjectorWrapper` 中转发 `on_host_pools_registered()`。
3. **vLLM MRV2 引用式 staging 直拷 UVA 缓冲池（`#63633`）**：`StagedWriteTensor` 不再将 `np.ndarray` / `array.array` 展开为 Python list 再调 `async_tensor_h2d`，而是直接保存 chunk 引用，在 `copy_to_uva` 中将一维连续缓冲直接拷入 Round-Robin UVA 池，消除额外 pinned staging 拷贝。
4. **vLLM KV Offload P2P 按拓扑模式拆分超时阈值（`#63781`）**：
   - 对称 P2P 模式下，lookup 探针已确认远端 peer 持有目标 KV 块，因此将加载超时从 `30.0s` 缩短至 `_P2P_LOAD_TIMEOUT_S = 10.0s`（`3.0x` 更快从卡死 peer 切回本地重算）；
   - 分离式 PD 模式下，prefiller 可能仍在排队计算，保留 `_PD_LOAD_TIMEOUT_S = 30.0s`；
   - 同时将 `_ABORT_ACK_TIMEOUT_S` 从 `10.0s -> 5.0s`（`2.0x`）、`_LOOKUP_TIMEOUT_S` 从 `30.0s -> 8.0s`（`3.75x`），并维持 `_REAPED_ID_RETENTION_S = 60.0s > 30.0s` 防止迟到完成消息被误判为未知请求。
5. **vLLM CUDA 13.4 `SMPartition` 局部性域 NVFP4 MoE 分配与 Step3 `<` 工具解析修复（`#61685`, `#60968`）**：引入 `LocalizedResources`（`SMPartition` + `GreenContext` + `LocalizedAllocator`），例如将 140 SM GPU 划分为 `[60, 68]` 两个 L2/SM 局部性计算域与 `12` 个尾块回填 SM（`VLLM_MOE_LOCALIZE_BACKFILL`），将 FlashInfer CuTeDSL NVFP4 MoE 的 `gemm1/gemm2` 权重固定在对应局部性域；同时将 Step3 `<steptml:parameter>` 正则从 `([^<]*)` 修复为非贪婪 `(.*?)` + `re.DOTALL`，保留含 `<` 的 SQL/HTML 参数。
6. **TensorRT-LLM 分离式 `/v1/responses` 无状态校验与 `MuseSparkOpenConfig` 转换（`#19883`, `#20643`）**：在 `OpenAIDisaggregatedServer` 中支持 `/v1/responses`，对多 worker 下无法跨进程共享的 `previous_response_id` 显式返回 HTTP `400`，并用 `_has_responses_output_event` 过滤开场事件（`response.created` / `response.in_progress`）；新增 `MuseSparkOpenConfig.from_open_format` 将逐层异构 `layer_types` 与 `layer_rope_theta` 规约为周期性 `every_n_layers_nope = 4`（32 层中 8 层全局 NoPE 注意力）。

---

## 2. 确定性实验验证汇总（`own-offline`）

运行 `python experiments/sglang_xeon_grok2_xpu_wan_vllm_uva_p2p_loc_trtllm_phase94/run.py --check` 结果如下：

| 框架与 PR | 关键指标 / 不变量 | 修复/优化前 | 修复/优化后 | 架构收益 |
|---|---|---:|---:|---|
| SGLang `#43341` / `#46049` | CPU 预分片 TP3 偏移 / Grammar 崩溃状态码 | `6144` / `200` | `0` / `500` | 消除越界切片与静默截断 |
| vLLM `#63633` MRV2 UVA | `int32`/`int64`/`float32` `array` 字节宽度 | 2 次 H2D 拷贝 | `4 / 8 / 4 B` 引用直拷 | 零中间 staging tensor |
| vLLM `#63781` P2P KV Offload | 对称 P2P Load / Lookup / Abort 超时 | `30 / 30 / 10 s` | `10 / 8 / 5 s` | `3.0x` / `3.75x` / `2.0x` 快速故障收敛 |
| vLLM `#61685` / `#60968` | 140 SM 局部性划分 (`[60, 68]`) / Step3 `<` 匹配数 | 未分区 / `0` 匹配 | `12` 尾块 SM / `2` 匹配 | L2 局部性绑定 + SQL `<` 无损解析 |
| TensorRT-LLM `#19883` / `#20643` | 分离式 `previous_response_id` / Muse Spark NoPE 层数 | `500` 异常 / 未支持 | HTTP `400` / `8/32` 层 | 无状态契约显式化 + 多模态配置归一 |
