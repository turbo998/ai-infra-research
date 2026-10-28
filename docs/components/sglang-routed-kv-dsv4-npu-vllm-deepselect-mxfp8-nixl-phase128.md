# Phase 128 深度研究（Track B）：SGLang 声明式路由 KV 存储栈、昇腾 NPU DeepSeek-V4 零同步解码调度与 vLLM BF16 DeepSelect 预填充索引器/ROCm MXFP8 融合/NIXL Push 模式

## 1. 背景与核心问题

现代推理引擎在支持混合注意力（Full + Sliding Window + Mamba 状态）、投机采样（EAGLE/EAGLE3/DSPARK）以及跨节点 PD 分离时，面临三大系统瓶颈：
1. **多变体统一内存池的组合爆炸**：为每种模型家族手写专用统一池、传输描述符与草稿模型绑定逻辑。
2. **单线程解码调度器中的主机-设备同步（Host-Sync）气泡**：在昇腾 NPU 上运行 DeepSeek-V4（DSV4）时，布尔掩码索引（`aclnnNonzeroV2`）与设备张量回读（D2H）阻塞单线程主机调度器长达数十毫秒。
3. **稀疏注意力（DSA）预填充索引器与 ROCm 微缩放量化的额外转换开销**：vLLM 在 `deep_select` 模式下仍将 BF16 索引器 logits 上采样为 FP32 再选 Top-K，而 ROCm 上的 MiniMax-M3.1 在 Gemma RMSNorm 后需额外启动一次 MXFP8 量化核并往返读写 BF16 激活。

本报告基于 2026-10-28 合入 `sgl-project/sglang`（6 个提交）与 `vllm-project/vllm`（3 个提交）的 9 个最新提交，建立完整的技术剖析与定量验证。

## 2. 关键实现路径与机制剖析

### 2.1 SGLang 5 连提交：声明式统一 KV 子池、存储计划传输与路由缓存视图（`5433093d83b8`, `298b56359664`, `f80cb3a2fd0f`, `240862923e8d`, `146378629d3e`）
1. **从层声明构建统一池（`5433093d83b8`）**：新增 `sglang/srt/mem_cache/kv/{decl.py,plan.py,build.py}`，每层声明几何布局（`MHAGeometry`/`MLAGeometry`/`StateGeometry`）与生命周期（`StateLifetime(rank=1)`、`FullLifetime(rank=2)`、`WindowLifetime(rank=3)`）。相同生命周期与分配单元的层共享同一子池，并按生命周期 Rank 在统一缓冲区中双端/浮动排布，统一收敛至 `build_unified_pools` 入口。
2. **基于存储计划构建 PD 缓存传输（`298b56359664`）**：`get_kv_transfer_buf_infos(pool, storage)` 直接从 `storage.regions(storage.owner)` 提取连续指针与长度，使 PD 节点将主子池注册为单一整体包络区域，消除按层拆分注册开销。
3. **以路由缓存替换专用统一池变体（`f80cb3a2fd0f`）**：通过 `resolve_window_id_space` 与 `layer_router` 统一访问 `IdSpaceKind.FULL` 与 `IdSpaceKind.SLIDING_WINDOW`，移除专用池子类分支。
4. **将融合投机草稿绑定至路由 KV 视图（`240862923e8d`）**：FlashAttention（FA3/FA4）与 AITER 后端通过路由视图直接读取全注意力或滑动窗口子池的转换页表，支持全量 `SpecMode`（EAGLE/EAGLE3/DSPARK/NGRAM）在统一内存池上运行。
5. **跨运行时消费者共享 `kv_storage_of` 接口（`146378629d3e`）**：将 PD 分离（Prefill/Decode）与 Beam Search（`BeamCoordinator`、`free_member_rows`）统一收敛至 `kv_storage_of(req_to_token_pool, token_to_kv_pool_allocator)`。

### 2.2 SGLang 昇腾 NPU DeepSeek-V4 解码调度链零主机同步改造（`f59cc7e45fd0`）
- **实现路径**：`python/sglang/srt/hardware_backend/npu/dsv4/{dsv4_allocator.py,dsv4_common_hooks.py}` 新增 `_compact_c4_loc` 并将 `maybe_write_dsv4_decode` 切换至 `batch.req_pool_indices_cpu`。
- **核心机制**：此前 DSV4-NPU 解码链存在 3 处设备同步点：① 投机验证包中的 `x[mask]` 触发 `aclnnNonzeroV2` 动态形状回读（`~3.9 ms/round`）；② 非投机分配路径 `_derive_c4_loc_from_full` 中的布尔索引继承整条队列排空等待（`32.5 ms/round`）；③ `maybe_write_dsv4_decode` 读取设备端 `req_pool_indices` 触发 D2H 同步（`31.4 ms/round`）。新实现通过设备端固定上界流水线 `cumsum -> masked_fill -> scatter`（无效槽位写入垃圾桶槽后切片丢弃，输出零填充至主机已知上界 `min(t, t//4 + bs)` 或 `t`）并直接读取主机镜像 `req_pool_indices_cpu`，将每步解码的主机同步次数从 **`3 -> 0`**，完全消除 `32.5 ms / 31.4 ms` 队列排空阻塞。

### 2.3 vLLM DSA 预填充索引器 BF16 DeepSelect 直接排序（`c105af429241`）
- **实现路径**：`vllm/model_executor/layers/sparse_attn_indexer.py`。
- **核心机制**：在启用 `sparse_indexer_topk_backend="deep_select"` 时，预填充阶段不再将 DeepGEMM 输出的 BF16 MQA 索引器 logits 转换为 FP32（省去 `as_fp32_logits`），而是直接传入 `deep_select_topk(..., end=row_ends)`。由于 BF16 向 FP32 拓宽是精确无损的，Top-K 选中值完全一致（GSM8K `0.924/0.923`，Passkey `20/20`）。在 GB200 上，单块 65,536 压缩键下的 Top-K 算子提速 **`3.7x–4.9x`**（128 行从 `66.5 -> 13.6 µs` **`4.9x`**，8192 行从 `1503.9 -> 410.7 µs` **`3.7x`**）；DeepSeek-V4.1-Flash DEP2 128K 前缀命中预填充中，`upcast + top-k` 耗时从 **`39.4 ms` 降至 `5.7 ms`（`6.912x`）**，整个索引器从 **`64.5 -> 30.9 ms`（`2.087x`）**，端到端步耗时降低 **`6.6%`**（`429.8 -> 401.5 ms`）。

### 2.4 vLLM NIXL `transfer_mode` Push/Pull 路由与 ROCm MiniMax-M3.1 融合 Gemma RMSNorm + MXFP8 量化（`066d9799347c`, `1cb31be70284`）
- **NixlConnector `transfer_mode`（`066d9799347c`）**：`KVConnectorFactory.get_connector_class` 支持在 `"kv_connector": "NixlConnector"` 下通过 `kv_connector_extra_config.transfer_mode`（`"pull"` 默认或 `"push"`）动态解析为 `NixlPullConnector` 或 `NixlPushConnector`（直接通过 `NIXL WRITE` 写入 Decode 预分配显存），对非法模式抛出 `ValueError`。
- **ROCm MiniMax-M3.1 融合 Gemma RMSNorm + MXFP8 量化（`1cb31be70284`）**：在 MI355X（CDNA4 `gfx950`）上将 `1x32` MXFP8 激活量化融合进 Gemma RMSNorm 尾声（Epilogue），直接向 `RocmDotScaledMxfp8LinearKernel` 传递携带行主序 E8M0 缩放的 `QuantizedActivation`。60 层解码器中有 59 层省去独立量化核启动与 BF16 往返读写，在 MI355X 上 `Norm + Quant` 提速 **`1.41x–2.06x`**，`Norm + Quant + qkv_proj GEMM` 提速 **`1.07x–1.19x`**，节省 **`1.9%`** 解码 GPU 内核总时间，AgentX P90 交互吞吐在 `c=16` 下提升最高 **`1.045x`**。

## 3. 确定性实验结果汇总

运行 `python experiments/sglang_routed_kv_dsv4_npu_vllm_deepselect_mxfp8_nixl_phase128/run.py`：

| 系统 / 提交 | 评估维度 | 优化前 / 基线 | 优化后 / 新路径 | 定量收益 |
|---|---|---:|---:|---|
| SGLang KV 栈 (`5433093d83b8`..`146378629d3e`) | 统一池构建与 PD 注册粒度 | 多套专用池 / 按层注册 | `build_unified_pools` / 1 个 Owner 包络区 | 统一全量 `SpecMode` 与 Beam/PD 存储接口 |
| SGLang NPU DSV4 (`f59cc7e45fd0`) | 解码调度链 Host-Sync 阻塞 | 3 次同步（`32.5 / 31.4 / 3.9 ms`） | `cumsum->masked_fill->scatter` + CPU 镜像 | **`0` 次同步**，消除 ~`32.5 ms` 队列排空等待 |
| vLLM DSA DeepSelect (`c105af429241`) | GB200 预填充索引器 Top-K / 端到端 | `39.4 ms` (FP32 upcast+topk) / `429.8 ms` | `5.7 ms` (BF16 DeepSelect) / `401.5 ms` | Top-K **`4.9x`**，Upcast+TopK **`6.912x`**，端到端 **`-6.6%`** |
| vLLM ROCm MXFP8 (`1cb31be70284`) | MI355X MiniMax-M3.1 Norm+Quant | 独立 Norm + `mxfp8_e4m3_quantize` | 融合 Norm Epilogue -> `QuantizedActivation` | Norm+Quant **`1.41x–2.06x`**，消去 59/60 层量化核 |
| vLLM NIXL (`066d9799347c`) | 仅支持 `NixlConnector` 的编排框架 | 无法通过统一名称切换 Push | `transfer_mode: "push"` -> `NixlPushConnector` | 零代码侵入启用 `NIXL WRITE` 直写传输 |
