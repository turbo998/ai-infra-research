# TensorRT-LLM DeepSeek-V4 Rubin (SM107) 原位 MXFP8 o_b_proj 与 CuteDSL MoE 异步输入量化、vLLM D2D 预热缓存播种与卸载失败指数退避、SGLang MiniMax M3.1 稀疏 NVFP4 注意力 (MSA-v1) 及推理厂商 Pareto 演进

> **English Version**: [trtllm-dsv4-rubin-vllm-d2d-seed-backpressure-sglang-msa-nvfp4.en.md](trtllm-dsv4-rubin-vllm-d2d-seed-backpressure-sglang-msa-nvfp4.en.md)  
> **关联问题 ID**: `trtllm-dsv4-rubin-vllm-d2d-seed-backpressure-sglang-msa-nvfp4` (Phase 74, Priority 83)  
> **关联可复现实验**: [`experiments/trtllm_dsv4_rubin_vllm_d2d_seed_backpressure_sglang_msa_nvfp4_sim/run.py`](../../experiments/trtllm_dsv4_rubin_vllm_d2d_seed_backpressure_sglang_msa_nvfp4_sim/run.py)  
> **证据分类**: `implementation` / `author-benchmark` (上游源码与 PR 实测) + `own-offline` (确定性标准库仿真器)

---

## 1. 背景与架构痛点

在 DeepSeek-V4 / DSpark 与 MiniMax M3.1 等新一代稀疏注意力 + MoE 模型的生产推理中，TensorRT-LLM、vLLM 与 SGLang 在 `2026-10-24` 至 `2026-10-25` 期间解决了六项关键性能与可靠性瓶颈：

1. **Rubin (`SM107`) 上 DeepSeek-V4 注意力输出投影拷贝开销与 RoPE 重复旋转**（[`TensorRT-LLM@d7e0ecac925d`](https://github.com/NVIDIA/TensorRT-LLM/commit/d7e0ecac925dc8f3fddfc2bff5ef5ab1d9149fbf)）：在 Rubin 架构上运行 DeepSeek-V4 MLA 时，`o_b_proj` MXFP8 线性层默认先分配临时张量再拷回预分配输出缓冲区；同时 `_quant_scale_qkv` 在首次 CUDA Graph 捕获期间才创建会触发非法 H2D 拷贝，且当 `q_b` GEMM 已内联完成 FP8 RoPE 时，`mla_rope_generation` 仍会重复改写 `q_pe`。
2. **CuteDSL MoE 输入异步量化在单流 Eager 模式下的主机分叉负优化**（[`TensorRT-LLM@032eb456ecb1`](https://github.com/NVIDIA/TensorRT-LLM/commit/032eb456ecb145ba10f9c9dc043cfaa8254d657c)）：将 `quantize_input_async` 分叉到辅助流与 `top-k` 路由 + `moe_sort` 重叠，在 GB300 `batch=8` CUDA Graph 下每层可省 `~25 us`（61 层共 `1.525 ms`），但在未开启多流或非捕获态的 Eager 模式下，主机端流切换与 Event 同步反而增加延迟。
3. **vLLM 预加载扩容副本冷启动重跑全量 Autotune 耗时过长**（[`vllm@94a29d316e8c`](https://github.com/vllm-project/vllm/commit/94a29d316e8c1d6dcf6ff75645c2e61c7e3c83fd)）：开启 `--warmup-level preload` 的新扩容或重启 Pod 若每次从零编译/调优 DeepGEMM、FlashInfer、Triton 与 PyTorch Inductor，冷启动长达 `~148 s`。
4. **vLLM KV 卸载存储层满盘 (`ENOSPC`) 时慢存储检测器失灵与抢占恢复漏存**（[`vllm@2b1f75c80f79`](https://github.com/vllm-project/vllm/commit/2b1f75c80f79496b313e28362f0856632dfbf5ea)、[`vllm@6f51433838b8`](https://github.com/vllm-project/vllm/commit/6f51433838b858a9cd0478b2ffee4d2c610a1712)）：当磁盘/对象存储（`fs`/`obj`）因 `ENOSPC` 瞬间报错返回时，完成耗时极短，导致基于延迟的 `EMABackpressureDetector` 误判存储健康并持续提交失败任务；此外请求被抢占重置回 0 tokens 后，`next_stored_chunk_idx` 未归零导致被驱逐的前缀块永远不再重新卸载。
5. **SGLang MiniMax M3.1 稀疏 NVFP4 注意力 (`MSA-v1`) 显存池与 FP8 舍入一致性**（[`sglang@4991bd33ae1c`](https://github.com/sgl-project/sglang/commit/4991bd33ae1c9d7468394eaf966d7db88f352b7b)）：MiniMax M3.1 采用 Q8KV4 + K-only FP8 稀疏索引（64 Q heads、4 KV heads、4 index heads、`block_size=128`、`topk_blocks=16`），若 `fused_gemma_qknorm_rope_fp8` 直接从 FP32 累加器转 FP8，会与先转 BF16 再转 FP8 的参考路径出现双舍入偏差。
6. **SGLang DeepSeek-V4 `FLASHINFER_MEGAMOE` 组合阶段共享专家漏加或重复累加**（[`sglang@ad950cdcd966`](https://github.com/sgl-project/sglang/commit/ad950cdcd9666a104321deeaf1cbf883b4e8c60b)）：`FLASHINFER_MEGAMOE` 后端自身完成跨 rank reduce（`is_complete=True`），但 `self.rows.group` 为 `None`，旧逻辑仅检查 `self.rows.group is not None` 导致 `summed=False`，使 `shared_output` 在非零 TP rank 上被跳过。

---

## 2. 核心实现机制与源码走读

### 2.1 TensorRT-LLM DeepSeek-V4 Rubin (`SM107`) 原位 `o_b_proj` 与 MoE 异步量化门控

- **原位 `o_b_proj` 输出与预注册 QKV Scale**（`d7e0ecac925d`）：在 `CuteDSLBlockScaledRubinLinear` 中增加 `allow_unpartitioned_output: bool = False`，使 `DeepseekV3Attention` 在 `SM107` 上将预分配的 `_rubin_dsv4_o_b_proj_out` 直接传入 `o_b_proj(..., out=self._rubin_dsv4_o_b_proj_out)`；在 `__init__` 预注册 `self._quant_scale_qkv = torch.ones(1, dtype=torch.float32, device="cuda")`；并在 `mla_rope_generation` 中透传 `q_rope_applied = fuse_q_fp8_in_rope && !q_pe.has_value()`，跳过冗余的 `q_pe` RoPE 写入。
- **MoE 异步输入量化多流门控**（`032eb456ecb1`）：将 `_should_quantize_input_async()` 严格限制为：
  $$\text{moe.comm is None} \land \neg\text{apply\_router\_weight\_on\_input} \land (\text{do\_multi\_stream}() \lor \text{is\_current\_stream\_capturing}())$$
  在 GB300 `batch=8` CUDA Graph 下每层节省 `25 us`（61 层节省 **`1.525 ms/step`**），同时在单流 Eager 模式下规避约 `0.58 ms/step` 的主机流切换开销。

### 2.2 vLLM D2D 预热缓存播种与卸载失败 EMA 反压

- **Daemon-to-Daemon (D2D) 缓存播种**（`94a29d316e8c`）：在 `vllm/Distributed/d2d_cache_seeding.py` 中，通过 `port = base_port + group_offset + rank` 建立点对点 TCP 通道（支持 `--seed-token` 鉴权），将已预热 Pod 的 `torch.compile`、Triton、DeepGEMM 与 FlashInfer 缓存打包传输给新 Pod，并由 `verify_cache_metadata()` 校验 GPU 型号、CUDA 与 cuDNN 版本，将扩容预热时间从 `148.0 s` 压缩至 **`3.2 s`（`46.25x`）**。
- **连续失败升压与抢占水位归零**（`2b1f75c80f79`、`6f51433838b8`）：`EMABackpressureDetector` 对非豁免介质（`fs`/`obj`，豁免 `p2p`）统计连续失败次数；当 `consecutive_failures >= 3` 时，将 EMA 抬升至 `_FAILURE_EMA_FACTOR * high_water_s = 4.0 * 0.050 = 0.200 s`，使 `ThrottledDropPolicy`（`ramp_factor=2.0`，满丢弃阈值 `0.100 s`）立即以 **`100%` 概率丢弃新卸载请求**，并通过空闲指数衰减自动探测恢复。同时在 `scheduler_integration.py` 中，当请求发生抢占（`num_computed_tokens == 0 and group_state.next_stored_chunk_idx > 0`）时将其重置为 `0`。

### 2.3 SGLang MiniMax M3.1 `MSA-v1` 稀疏 NVFP4 注意力与 DSv4 MegaMoE 修正

- **`MiniMaxSparseNvfp4AttnBackend` (`MSA-v1`)**（`4991bd33ae1c`）：每个 KV head 每 token 仅占 `144 B`（NVFP4 K+V 含 FP8 块缩放）+ `132 B`（K-only FP8 索引）= `276 B`（4 个 KV head 共 `1,104 B/token`，较 BF16 `2,048 B` 缩减 **`1.855x`**）。在 `131,072` 上下文解码时，每个 KV head 仅读取 `16 × 128 = 2,048` 个选中 token 的 NVFP4 KV 与 `1,024` 个块索引，总读取流量从稠密 BF16 的 `268.44 MB` 降至 **`1.72 MB`（带宽需求降低 `156.04x`）**。
- **BF16 中间舍入转 FP8 与 MegaMoE 组合判据**（`4991bd33ae1c`、`ad950cdcd966`）：`fused_gemma_qknorm_rope_fp8` 改为 `cast<fp8_e4m3_t>(o_bf16)` 保证与 `bf16.to(float8_e4m3fn)` 逐位一致；`MoeTransport.prepare_output` 增补 `or (self.rows.group is None and self.rows.post_experts_output_is_complete(is_tp_path=False))`，使 `FLASHINFER_MEGAMOE` 正确执行本地共享专家相加。

---

## 3. 架构师定量对比与 6 厂商 Pareto 前沿

运行 `python experiments/trtllm_dsv4_rubin_vllm_d2d_seed_backpressure_sglang_msa_nvfp4_sim/run.py --check` 验证：

| 推理平台 / 架构配置 | $/1M Tokens (USD) | P99 TTFT (ms) | 单用户 Decode TPS | 核心架构特征 (Phase 74 更新) |
|---|---|---|---|---|
| **NVIDIA Vera Rubin VR200 (NVL72, SM107)** | **$0.09** | `175.0` | `148.0` | 原位 MXFP8 `o_b_proj` + 预注册 QKV scale + 免重复 `q_pe` RoPE |
| **NVIDIA Blackwell Ultra GB300 (NVL72, SM103)** | **$0.14** | `215.0` | `118.0` | CuteDSL MoE 多流异步输入量化 (`-1.525 ms/step`) + QSA 持久化遍历 |
| **Fireworks AI (Blackwell Disagg + D2D Seed)** | **$0.18** | `240.0` | `106.0` | D2D 预热缓存跨 Pod 播种 (`46.25x` 扩容提速) + 满盘 EMA 熔断 |
| **Together AI (Turbo Spec + MSA-v1 NVFP4)** | **$0.19** | `250.0` | `102.0` | MiniMax M3.1 `MSA-v1` Q8KV4 稀疏注意力 (`156.04x` 128K 流量缩减) |
| **Groq LPU V2 (SRAM 确定性流水线)** | **$0.58** | `85.0` | `310.0` | 片上 SRAM 零 HBM 抖动，适合短上下文极低延迟交互 |
| **Cerebras WSE-3 (晶圆级 Swarm)** | **$0.62** | `92.0` | `420.0` | 晶圆级超大单步吐字率，受限于片上容量与多租户成本 |

---

## 4. 反证、边界条件与后续可证伪问题

1. **为什么 `EMABackpressureDetector` 必须把 `p2p` 介质加入 `_FAILURE_EXEMPT_MEDIA`？**  
   P2P 远程节点传输失败通常源于单个对端实例抖动或重连，而非全局存储池满盘；若对 `p2p` 触发全局 100% 丢弃反压，会误伤整个集群其它健康 P2P 节点的 KV 卸载。
2. **后续可证伪问题**：在 MiniMax M3.1 `MSA-v1` 的 `16 × 128 = 2,048` 稀疏选中窗口下，若开启 DSpark 投机解码（一次验证 `K=4` 个草稿 token），`MSAV1DecodeAttentionPlan` 的共享选中块率在真实代码生成负载下能否保持在 `85%` 以上？
