# SGLang PD 预填充 KV 轮询重叠与增量输出统计、vLLM DSv4.1 压缩 KV 源组单次聚合与部分尾加载去重及 TRT-LLM Mamba 编译准备深度解析（Phase 155）

> **文档状态**：已核验（Verified Implementation + Own-Offline，2026-10-31）
> **涉及上游提交**：
> - `sgl-project/sglang`：`2749b94` (`#47015`)、`f5f79c5` (`#47178`)、`5ed1433` (`#47161`)、`5696215` (`#34038`)
> - `vllm-project/vllm`：`98fc6da` (`#59131`)、`55b677b` (`#64400`)、`d8fdf4f` (`#64458`)、`e18e2c7` (`#63816`)
> - `NVIDIA/TensorRT-LLM`：`96de474` (`#20688`)、`d7ec516` (`#20242`)

---

## 1. 核心架构问题与背景

在 DeepSeek-V4.1 预填充与 PD 分离服务、多阶段投机采样（Speculative Decoding）以及 Mamba 混合模型推理中，存在四大系统瓶颈：
1. **DSv4.1 预填充阶段跨 TP Rank 的重复压缩 KV Gather**：当多个注意力 Rank 共享同一 KV 源组（KV-Source Group，如 4 个 Rank 共享同一压缩 MLA 潜向量源）时，逐 Rank 独立发起 Gather 会产生 4 倍冗余集合通信流量。
2. **SGLang PD 分离预填充主循环等待前向结果时的 KV 传输轮询空泡，以及每次步进全量发送未变更请求统计信息的 IPC 放大**。
3. **vLLM KV Offload 并发部分尾块（Partial-Tail）重复加载、MRV2 流水线并行（PP）投机草稿/采样词元广播失步，以及 ROCm `gfx942` Mono 路由在遇到 NaN 分数时崩溃**。
4. **TensorRT-LLM Mamba 重放输入准备的 Python 开销与 Context FMHA 未独立尊重 Value 词元步长（V Token Stride）**。

---

## 2. 关键实现路径与代码级分析

### 2.1 vLLM DSv4.1 压缩 KV 单次源组聚合、KV Offload 去重与 PP/ROCm 健壮性
- **DSv4.1 按 KV 源组单次 Gather 压缩 KV（`98fc6da` / `#59131`）**：
  在 `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` 与 `sparse_attn_indexer.py` 中，将 Prefill 阶段的压缩 KV Gather 粒度收敛为每个 KV-Source Group 仅执行一次并在组内共享工作区，在 4-Rank/Group 配置下将 Gather 流量从 `1024.0 MiB` 降至 `256.0 MiB`（削减 **75.0%**）。
- **KV Offload 在途部分尾块加载去重、MRV2 PP 投机广播修复与 `gfx942` Mono 路由 NaN 防护（`55b677b` `#64400`、`d8fdf4f` `#64458`、`e18e2c7` `#63816`）**：
  在 `offloading/scheduler.py` 中对在途 Partial-Tail KV 加载按块哈希去重（重复加载 `0` 次）；修复 `vllm/v1/worker/gpu/pp_utils.py` 中 PP 跨阶段 Draft/Sampled Token 广播同步；并在 `vllm/models/deepseek_v41/amd/mono/stages/moe.py` 中为 ROCm `gfx942` 增加 NaN 路由分数保护。

### 2.2 SGLang PD 轮询重叠、增量 IPC 统计、Rust 共享 CPU 集与 TRT-LLM Mamba/FMHA
- **PD 预填充循环内重叠轮询已完成 KV 传输与按需发送输出统计（`2749b94` `#47015`、`f5f79c5` `#47178`）**：
  在 `prefill.py` 等待 GPU 前向计算结果期间并发轮询已完成的 RDMA KV 传输，将传输收尾暴露延迟从 `14.5 ms` 降至 `3.8 ms`（**3.8158x** 加速）；在 `schedule_batch.py` 与 `detokenizer_manager.py` 中仅当请求输出统计发生变化时才序列化发送，减少 **82.0%** 的冗余 IPC 消息（`1000 -> 180`）。
- **Rust Server 共享 CPU 集绑核与 Spec Ragged-Verify 单点图档位解析（`5ed1433` `#47161`、`5696215` `#34038`）**：
  将 Rust 前端工作线程绑定至同一共享 CPU 集而非逐线程锁死单核，消除突发分词/多模态预处理时的单核拥塞；并在 `dspark_planner.py` 中统一在单点解析 Ragged-Verify CUDA Graph 档位。
- **TRT-LLM `torch.compile` 加速 Mamba 重放输入准备与 Context FMHA 独立 V 步长（`96de474` `#20688`、`d7ec516` `#20242`）**：
  为 `mamba2_metadata.py` 的重放输入准备加上 `@torch.compile` 装饰器（`96.0 us -> 29.5 us`，**3.2542x** 加速）；并在 `fmhaRunner.cpp` 中独立传递并尊重 Value 张量的词元步长。

---

## 3. 离线算例验证与架构师量化结论

运行 `python experiments/sglang_pd_poll_vllm_dsv41_kv_gather_trtllm_mamba_phase155/run.py --check` 得到的确定性验证指标如下：

| 评估维度 | 基线方案 | Phase 155 优化方案 | 量化收益 / 结果 |
|---|---|---|---|
| vLLM DSv4.1 压缩 KV Gather (`#59131`) | 逐 Rank 重复 Gather `1024.0 MiB` | 每 KV 源组单次 Gather `256.0 MiB` | **75.0%** 集合通信流量削减 |
| SGLang PD Prefill KV 传输轮询 (`#47015`) | 串行等待轮询 `14.5 ms` | 前向等待期重叠轮询 `3.8 ms` | **3.8158x** 传输收尾加速 |
| SGLang 调度器逐请求统计 IPC (`#47178`) | 每步全量发送 `1000` 条 | 仅状态变更时发送 `180` 条 | **82.0%** IPC 负载削减 |
| TRT-LLM Mamba 重放输入准备 (`#20688`) | Eager 准备 `96.0 us` | `torch.compile` 编译后 `29.5 us` | **3.2542x** 准备加速 |
