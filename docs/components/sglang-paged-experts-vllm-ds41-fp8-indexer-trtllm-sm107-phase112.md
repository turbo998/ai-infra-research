# Phase 112 Track B：SGLang Paged-Experts 单次同步驻留、vLLM 提示词打分前缀缓存边界与 DSv4.1 FP8 稀疏索引器、TensorRT-LLM Rubin SM107 MiniMax-M3（2026-10-28）

> **英文对照 / English Version**: [`sglang-paged-experts-vllm-ds41-fp8-indexer-trtllm-sm107-phase112.en.md`](./sglang-paged-experts-vllm-ds41-fp8-indexer-trtllm-sm107-phase112.en.md)  
> **7-Card 4:5 视觉评测报告**: [`../reports/ai-infra-phase112-benchmark-cards.html`](../reports/ai-infra-phase112-benchmark-cards.html)  
> **可复现实验目录**: [`../../experiments/sglang_paged_experts_vllm_ds41_fp8_indexer_trtllm_sm107_phase112/`](../../experiments/sglang_paged_experts_vllm_ds41_fp8_indexer_trtllm_sm107_phase112/)

---

## 1. 证据分级与一手代码定位（10 项推理调度、KV Cache 与多模态提交）

| # | 仓库与 Commit | 日期 | 证据等级 | 关键代码路径与核心修复/优化机制 |
|---|---|---|---|---|
| 1 | [`sglang@a4a19332ab`](https://github.com/sgl-project/sglang/commit/a4a19332abd9b15bdfc2b983e0032b54f687af04) | 2026-10-27 | `implementation` | `python/sglang/srt/layers/moe/paged_experts/{executor,residency,runners}.py`：在设备端用 `scatter_add_` 统计各专家路由条目数 `counts`，并与 `slot_expert`、`slot_lastuse`、`step` 拼接后单次 `.tolist()` 拷回主机，将每层 D2H 同步从 4 次压缩至 1 次 |
| 2 | [`sglang@69767e5aa6`](https://github.com/sgl-project/sglang/commit/69767e5aa6f2da1eaf135e28f4a46e45c6f35be2) | 2026-10-27 | `implementation` | `python/sglang/multimodal_gen/runtime/utils/camera_geometry.py`：`get_plucker_embeddings` 通过 `torch.promote_types(c2ws_mat.dtype, torch.float32)` 将相机位姿与像素网格提升至 FP32，修复 BF16 在坐标 $>256$ 时相邻像素坍缩至同一射线的问题 |
| 3 | [`sglang@c45ebb507d`](https://github.com/sgl-project/sglang/commit/c45ebb507d6e02daa66bef22aef2daf177a13800) | 2026-10-27 | `implementation` | `python/sglang/multimodal_gen/runtime/{models,pipelines}/hunyuan_image3.py`：修复 CUDA `FusedMoE` 原地覆写输入导致 `shared_mlp` 读取错误特征的问题，并将 `FlowMatchEulerDiscreteScheduler` 设为 `use_uniform_sigmas=True` 消除双重 shift |
| 4 | [`vllm@e1e77a279d`](https://github.com/vllm-project/vllm/commit/e1e77a279d68a9601be062cb267a5f9911c422c7) | 2026-10-27 | `implementation` | `docs/training/prompt_token_id_logprobs.md`、`rust/src/text/src/lower/logprobs.rs`：当请求指定 `prompt_logprob_start` 且 `skip_reading_prefix_cache=False` 时，允许按块对齐复用 `prompt_logprob_start` 之前的本地前缀缓存，同时对外部 KV Connector 强制 `skip_reading_external_kv=True` |
| 5 | [`vllm@a2ad4b74b9`](https://github.com/vllm-project/vllm/commit/a2ad4b74b9f76e99eae3148b729993ba16d474ff) | 2026-10-27 | `implementation` | `vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py`、`vllm/model_executor/layers/sparse_mqa_indexer.py`：在 SM100 上为 DeepSeek-V4.1 的 **FP8 索引器缓存**开启 DeepGEMM 稀疏 MQA logits 内核（将每行 2 的幂次缩放存为 4 个相同 UE8M0 字节），默认自动开启 (`indexer_sparse_logits=None`) |
| 6 | [`vllm@2897401370`](https://github.com/vllm-project/vllm/commit/289740137027e1d91490fcf9a48490a8a712e200) | 2026-10-27 | `implementation` | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`：新增 NumPy 向量化 `_contiguous_runs(src_ids, dst_ids)` 并在同一 KV group 的区域间缓存连续段划分，加速 Mooncake KV 传输描述符构建 |
| 7 | [`vllm@22d1bc8814`](https://github.com/vllm-project/vllm/commit/22d1bc881401041310f1c4808f1da64cdf281571) | 2026-10-27 | `implementation` | `rust/src/server/src/routes/openai/{chat_completions,completions}.rs`：vLLM Rust Server 在携带 `finish_reason` 的终态 SSE chunk 中流式输出 `kv_transfer_params`，与 Python 前端保持严密一致 |
| 8 | [`vllm@05a303fdc8`](https://github.com/vllm-project/vllm/commit/05a303fdc8681f0dd586a8ac22103f3b4a8bf4ad) | 2026-10-27 | `implementation` | `rust/src/server/src/routes/checkpoint.rs`、`rust/src/engine-core-client/src/protocol/utility.rs`：新增进程级 Checkpoint HTTP 生命周期接口 (`/checkpoint/prepare`、`/checkpoint/status`、`/checkpoint/restore`) 与 5 态状态机 |
| 9 | [`TensorRT-LLM@854ccee8a9`](https://github.com/NVIDIA/TensorRT-LLM/commit/854ccee8a9f0dded7235a0642032de105b3b47f7) | 2026-10-23 | `implementation` | `tensorrt_llm/_torch/attention/backends/{fmha/flashinfer_trtllm_gen,sparse/minimax_m3/cache_manager}.py`：在 Rubin (`SM107`) 上启用 MiniMax-M3（稀疏 MSA 层豁免保留 NVFP4 KV，共享 Eagle3 草案层通过 `FlashInferTrtllmGenFmha` 运行 128-token 分页，KV 池零初始化） |
| 10 | [`TensorRT-LLM@b0af321b0c`](https://github.com/NVIDIA/TensorRT-LLM/commit/b0af321b0c8771291bf201acf8bc2e68586ee5d9) | 2026-10-23 | `implementation` | `tensorrt_llm/_torch/modules/low_m_gemm.py`：在 `torch.cuda.is_current_stream_capturing()` 期间绕过 `_output_buffers` 缓存并为每次调用分配独立输出张量，防止单图内多次调用同一模块（如投机解码草案 logits）相互覆写 |

---

## 2. 核心架构机制与数学形式化深挖

### 2.1 SGLang Paged-Experts 单次 D2H 同步与 BF16 相机射线坍缩分析 (`sglang@a4a19332ab`, `sglang@69767e5aa6`)

1. **Paged-Experts D2H 同步合并 (`a4a19332ab`)**：在主机规划的 Paged-Experts 步中，旧代码依次调用 `torch.unique(topk_ids).tolist()`、`slot_expert.tolist()`、`slot_lastuse.tolist()` 以及 `store_from` 中的 `self.step.item()`，每层触发 $4$ 次 D2H 流同步。新实现先在 GPU 上用 `scatter_add_` 计算长度为 $E+1$ 的路由频次向量 `counts`（槽位 `-1` 映射至索引 $E$），再通过单次拼接拷贝：

$$\text{host\_vec} = \text{concat}\big([\text{slot\_expert}_{[K]},\ \text{slot\_lastuse}_{[K]},\ \text{step}_{[1]},\ \text{counts}_{[E+1]}]\big).\text{tolist}()$$

将每层 D2H 同步从 $4$ 次降至 $1$ 次（61 层模型每步减少 $183$ 次同步，按 $14\text{ }\mu\text{s}$/次计节省 **$2562\text{ }\mu\text{s}$**）。
2. **Plücker 射线 BF16 坐标坍缩 (`69767e5aa6`)**：BF16 尾数仅 7 位（含隐含位共 8 位精度），因此整数坐标在 $(256, 512]$ 区间步长为 $2$，在 $(512, 1024]$ 区间步长为 $4$。对于宽度 $W = 640$ 的帧，BF16 网格仅能表示 $257 + 128 + 31 = 416$ 个不同列坐标，导致 **$224$ 列（$35\%$）** 相邻像素坍缩到完全相同的相机射线上；强制提升至 `torch.float32` 恢复了全部 $640$ 列的精确几何射线。

### 2.2 vLLM 提示词打分前缀缓存边界与 DeepSeek-V4.1 FP8 稀疏 MQA 索引器 (`vllm@e1e77a279d`, `vllm@a2ad4b74b9`)

1. **提示词打分前缀缓存复用上界 (`e1e77a279d`)**：若请求要求从位置 $s = \text{prompt\_logprob\_start}$ 开始对提示词打分（即计算位置 $s+1 \dots L-1$ 的对数概率），则前 $s$ 个未打分行在开启 `skip_reading_prefix_cache=False` 时可安全复用本地块对齐缓存：

$$N_{\text{reusable}} = \min\!\left(N_{\text{cached}},\ \left\lfloor \frac{s}{B} \right\rfloor \times B\right)$$

当 $B=4, s=5$ 时，前 $4$ 个 token 直接命中本地缓存，而外部 KV Connector 因无法保证部分打分行的契约被显式设 `skip_reading_external_kv = True`。
2. **DeepSeek-V4.1 SM100 FP8 索引器缓存的 UE8M0 打包 (`a2ad4b74b9`)**：DeepGEMM 稀疏 MQA logits 内核要求粒度为 32 的打包 UE8M0 缩放因子。SM100 FP8 索引器缓存对每行采用单一 2 的幂次缩放 $2^{\lceil \log_2(\text{amax}/448) \rceil}$，并将其存储为 **4 个相等的 UE8M0 字节**（打包为一个 `int32`），从而无需转换即可直接复用 `fp8_fp4_sparse_mqa_logits` + `DeepSelect` top-$k$ 管线，使索引复杂度从 $O(L)$ 降至 $O(N_{\text{cand}}\cdot B_{\text{cand}})$（在 $64\text{K}$ 上下文、$64\times 64=4096$ 候选 token 下减少 **$16\times$** 计算量）。

### 2.3 TensorRT-LLM Rubin SM107 MiniMax-M3 分层路由与 Low-M GEMM 图捕获防别名 (`854ccee8a9`, `b0af321b0c`)

- **Rubin SM107 MiniMax-M3 异构注意力路由 (`854ccee8a9`)**：在 `sm_107` 上，稀疏目标层豁免保留 `NVFP4` KV Cache（由 MSA 与 Triton 稀疏解码服务），稠密目标层保留 `FP8` + SDPA，而共享 Eagle3 草案层采用 `128`-token 页面并通过 `FlashInferTrtllmGenFmha`（`EXTRA_PAGE_SIZES_ONLY_SMS = {107}`）执行；同时所有 KV 池在首次请求前强制零初始化，防止 CSR 稀疏预填充或尾页读取未初始化字节。
- **Low-M GEMM CUDA Graph 捕获输出隔离 (`b0af321b0c`)**：`low_m_gemm` 按 `(module_key, m_bucket, n)` 缓存输出缓冲区。在 CUDA Graph 捕获期间，若同一模块（如投机解码中的 `lm_head`）被连续调用多次且调用方保留了早期返回的张量引用，复用同一缓存地址会导致早期草案步的 logits 被最后一步覆写；修复后在 `is_capturing=True` 时每次调用分配独立 `torch.empty((m, n))`。

---

## 3. 本地确定性复现实验结果 (`own-offline`)

运行 `python experiments/sglang_paged_experts_vllm_ds41_fp8_indexer_trtllm_sm107_phase112/run.py --check` 得到的密封结果（[`summary.json`](../../experiments/sglang_paged_experts_vllm_ds41_fp8_indexer_trtllm_sm107_phase112/summary.json)）：

| 检验维度 | 旧路径 / 基线表现 | 修复/优化后表现 | 关键量化指标 |
|---|---|---|---|
| SGLang Paged-Experts 每步同步数 (61 层) | `244` 次 D2H 同步/步 | `61` 次 D2H 同步/步 | 同步次数降低 **`4.0x`**，每步节省 **`2562 µs`** |
| SGLang Plücker 射线像素网格 ($W=640$) | BF16 仅 `416` 个独立列 | FP32 完整 `640` 个独立列 | 消除 **`224` 列 (`35%`)** 相邻射线坍缩 |
| vLLM 提示词打分前缀缓存 ($B=4, s=5$) | 全部跳过 (`0` tokens) 或返回 `None` | 本地缓存命中 `4` tokens，外部连接器 `0` | 兼顾前缀缓存复用与部分提示词打分正确性 |
| vLLM DSv4.1 FP8 稀疏 MQA 索引器 ($64\text{K}$) | 稠密打分 `65,536` tokens | 候选块打分 `4,096` tokens (`0x7b7b7b7b` UE8M0) | 索引器 logits 计算量降低 **`16.0x`** |
| TRT-LLM Low-M GEMM 图捕获连续 3 步调用 | 别名覆写为 `[33.0, 33.0, 33.0]` | 独立缓冲区保留 `[11.0, 22.0, 33.0]` | 彻底消除捕获图内多次调用的输出互相覆写 |

---

## 4. 解决方案架构师行动指南与可证伪问题

1. **架构师检查清单**：
   - 在 Rubin (`SM107`) 无外网生产集群部署 TensorRT-LLM MiniMax-M3 + Eagle3 (`tokens_per_block=128`) 时，必须预先在联网节点设置 `FLASHINFER_CUBIN_DIR` 预热 128-token FMHA cubin，再拷贝至生产节点并设置 `FLASHINFER_NO_DOWNLOAD=1`，且不得从 `TLLM_FMHA_LIBS` 中剔除 `flashinfer_trtllm_gen`。
   - 使用 vLLM `--enable-sleep-mode` 配合 CRIU/外部进程快照时，编排器应先调用 `POST /checkpoint/prepare?mode=wait`，轮询 `GET /checkpoint/status` 直至返回 HTTP `200` (`state="prepared"`) 后再执行快照。
2. **后续可证伪问题**：
   - 当 DeepSeek-V4.1 开启 `decode_context_parallel_size > 1` 时，`SparseMQAIndexer` 为何回退至稠密路径，跨 rank 候选块索引聚合的通信开销边界在哪里？
