# SGLang gfx950 MTP 全 CU 切分、DSpark 无主机序列长度元数据、Engram 分层表、EPLB 融合计数与 vLLM GLM-5.3 TMA MoE 加速（Phase 117 / Priority 126）

> **证据分类**：Upstream `implementation` / `author-benchmark`（10 个上游公开提交，2026-10-27）+ 本仓库 `own-offline` 确定性调度与吞吐建模（`experiments/sglang_vattn_dspark_engram_moe_eplb_vllm_tma_kda_phase117/run.py`）。

## 1. 核心架构问题与背景

在 DeepSeek-V4 / V4.1（DSpark / Engram / DSA）、Kimi-K3（KDA + MLA）与 GLM-5.3 / Qwen3.5 的生产推理运行时中，SGLang 与 vLLM 近期解决了以下关键运行时瓶颈：

1. **小批量 MTP 验证在 AMD MI350X (`gfx950`, 256 CUs) 上的 CU 饥饿**：当请求并发较小（如 `B=8, H_kv=4`）时，未切分的 `vattn-asm` MTP 验证注意力仅激活 `32` 个 CU（利用率 `12.5%`），其余 `224` 个 CU 空闲。
2. **DSpark 投机采样元数据构建中的 Host 序列长度同步阻塞**：在构建 DSpark 草稿块元数据时，若依赖主机侧 `seq_lens` 切片与 `.item()` 同步，会在每个投机步引入数十微秒的 D2H 同步气泡。
3. **EPLB（专家并行负载均衡）在 CUDA Decode Graph 中的多内核计数开销与 MegaMoE 分块失配**：CUDA Graph 解码路径中分立的专家映射与计数记录内核增加了图节点启动延迟；而 MegaMoE 若仅将本卡 `local_tokens` 而非 EP 全局 `global_tokens` 传入分块启发式，会在 All-to-All 展开后选出过小的 `M-tile`。
4. **vLLM GLM-5.3 MoE Up/Down 投影内存搬运瓶颈与 Kimi-K3 KDA Prefill 工作区未预留风险**：GLM-5.3 MoE 投影缺乏硬件 TMA 异步张量加载；Kimi-K3 若在显存 Profiling 之后才分配 KDA Prefill 工作区（约 `384 MiB`），会侵占已分配给 KV Cache 的显存池并触发运行期 OOM。

---

## 2. 上游一手实现剖析（10 个零重叠 Commit）

| # | 仓库与提交 | 日期 | 核心实现路径与运行时机制 |
|---|---|---|---|
| 1 | `sgl-project/sglang@2c83dc6656` (`#45555`) | 2026-10-27 | `python/sglang/kernels/ops/attention/vattn_asm_gfx950/__init__.py`：在 AMD `gfx950`（256 CUs）上对小批量 MTP verify 注意力按 KV 序列维度做全 CU 切分（`32 -> 256` 活跃 CU，CU 覆盖率提升 `8.0x`）。 |
| 2 | `sgl-project/sglang@a0d4034d5b` (`#46891`) | 2026-10-27 | `python/sglang/srt/speculative/ragged_verify.py` & `deepseek_v4_backend.py`：完全在设备侧构建 DSpark 草稿块元数据，移除主机侧序列长度读取（4 步投机累计节省 `68.0 us` 主机同步开销）。 |
| 3 | `sgl-project/sglang@90a7339419` (`#46893`) | 2026-10-27 | `docs/docs/advanced_features/engram.mdx`：DeepSeek-V4.1 支持通过 `--engram-config` 显式配置 Engram 记忆表驻留位置（HBM vs 主机锁页内存分层卸载，24 GiB 表卸载 75% 可释放 `18.0 GiB` HBM）。 |
| 4 | `sgl-project/sglang@f9f2665b6e` (`#44797`) | 2026-10-27 | `python/sglang/srt/layers/moe/topk.py`：在 `select_experts` 的 AITER 与 `radix4` 分发分界线两侧统一保持 `fp32` 路由修正偏置（correction bias），消除 `radix4` 批次阈值附近的偏置降精度跳变（消除 `1.4%` 边界专家翻转）。 |
| 5 | `sgl-project/sglang@7e142dd9fd` (`#46093`) | 2026-10-27 | `python/sglang/srt/layers/moe/mega_moe.py`：始终向 Mega MoE 分块启发式传入全局 Token 数（`global_tokens`），避免 EP8 下用局部 Token 数误选 `M=16` 小分块。 |
| 6 | `sgl-project/sglang@baa349af7b` (`#46497`) | 2026-10-27 | `python/sglang/kernels/ops/moe/eplb_map_record.py`：在 CUDA Decode Graph 中将专家映射与激活计数融合为单一内核（从 3 个内核节点压缩为 1 个，净减 `2` 个图节点）。 |
| 7 | `sgl-project/sglang@3145c7ae66` (`#43839`) | 2026-10-27 | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`：在 AMD 后端支持将 Qwen3.5 的 BF16 GDN 线性投影在线量化为 `MXFP4`（有效位宽 `4.25 bits/weight`，权重体积压缩 `3.7647x`）。 |
| 8 | `sgl-project/sglang@7f07519943` (`#43271`) | 2026-10-27 | `python/sglang/srt/layers/attention/linear/kda_backend.py` & `dsa_indexer_kpool.py`：在混合注意力（DSA/KDA）MTP 验证路径中正确屏蔽 MoE Padding Token（如 `36` 真实 Token 补齐至 `64` 时的 `28` 个 Padding Token），防止污染 KDA 递归状态与 DSA 索引池。 |
| 9 | `vllm-project/vllm@c56706e88e` (`#63820`) | 2026-10-27 | `vllm/model_executor/layers/fused_moe/fused_moe.py`：在 MoE Up/Down 投影中启用硬件 TMA 异步张量加载，上游作者实测（`author-benchmark`）在 GLM-5.3 上取得 **3.1%–3.4%** 端到端吞吐提升（模型取中位 `3.3%`）。 |
| 10 | `vllm-project/vllm@e96b490b12` (`#63989`) | 2026-10-27 | `vllm/models/kimi_k3/nvidia/kda.py`：在 vLLM 显存 Profiling 阶段之前预留 Kimi-K3 的 KDA Prefill 工作区（`384 MiB`），将未入账显存溢出降至 `0.0 MiB`。 |

---

## 3. 离线模型验证汇总（`run.py --check`）

- `gfx950_cu_utilization_gain_ratio`: `8.0`（小批量 MTP 验证活跃 CU 从 `32` 扩展至全部 `256` 个 CU）
- `dspark_step_overhead_saving_us`: `68.0`（消除 DSpark 草稿块元数据主机同步开销）
- `eplb_decode_graph_node_reduction`: `2`（CUDA Decode Graph 中 EPLB 记录从 3 节点融合为 1 节点）
- `gdn_mxfp4_weight_compression_ratio`: `3.7647`（BF16 `16 bits` -> MXFP4 `4.25 bits` 含 E8M0 块缩放）
- `vllm_glm53_tma_e2e_gain_pct`: `3.3`（对应上游报告的 `3.1%–3.4%` 端到端吞吐提升区间）
- `reserved_kv_budget_overrun_mib`: `0.0`（Kimi-K3 KDA Prefill 工作区前置预留后零预算透支）
