# FlashInfer SDSA 词元稀疏 MQA 解码与 VSA/MiniMax-H3/NVFP4 算子、Triton Gluon/NVPTX TTI 向量加载与 TileLang SM120 块缩放 GEMM 深度解析（Phase 155）

> **文档状态**：已核验（Verified Implementation + Own-Offline，2026-10-31）
> **涉及上游提交**：
> - `flashinfer-ai/flashinfer`：`fd9893c` (`#7849`)、`ce6dc7d` (`#7836`)、`4c3d69a` (`#7850`)、`5708bb6` (`#6420`)、`c4ea174` (`#7862`)
> - `triton-lang/triton`：`72bcaf2` (`#12400`)、`f37fda8` (`#12337`)
> - `tile-ai/tilelang`：`7a5f446` (`#3257`)、`356f309` (`#3276`)、`134ade7` (`#3278`)

---

## 1. 核心架构问题与背景

在 DeepSeek 稀疏注意力（DSA/SDSA）、Hopper/Blackwell 变长块稀疏注意力（VSA）以及 SM100/SM120 块缩放 FP4/FP8 编译降级中，存在以下关键瓶颈：
1. **词元稀疏 MQA 解码缺少原生 `trtllm-gen` 高吞吐内核**：在 Token-Sparse Attention 中，每个 Query Token 仅关注由 DSA Indexer 选出的 Top-K Key 子集（如 32K 上下文中选 2,048 个 Key），若回退稠密 MQA 会产生 16 倍冗余 KV 访存。
2. **Hopper `cake_vsa_sm90` 长对模式队列失衡与 MiniMax-H3 B200 尾波次开销**：单队列 GQA 程序在处理长 Pair-Mode 块稀疏计划时存在负载倾斜；MiniMax-H3 变长注意力在 B200 上缺少 8-CTA 协同量化与尾块调度。
3. **行跨步（Row-Strided）BF16 输入破坏 NVFP4 Cluster-Split 计划与 SM100 Split-K 重复读取缩放因子 `alpha`**。
4. **Triton Gluon 累加器未释放与 LLVM NVPTX TTI 误拆解包向量加载，以及 TileLang SM120 奇数 Warp 网格与分区异步拷贝降级缺陷**。

---

## 2. 关键实现路径与代码级分析

### 2.1 FlashInfer SDSA、VSA SM90、MiniMax-H3 与 Blackwell FP4
- **`trtllm-gen` 原生 Token-Sparse MQA Decode（`5708bb6` / `#6420`）**：
  在 `csrc/trtllm_fmha_kernel_launcher.cu` 与 `flashinfer/decode.py` 中新增 SDSA 词元稀疏 MQA 解码路径，使每个 Query 直接按稀疏索引表拉取 Top-K KV 页，在 32K 上下文 Top-2048 场景下将解码耗时从 `192.0 us` 降至 `28.0 us`（**6.8571x** 加速）。
- **Hopper `cake_vsa_sm90` 双队列规划路由与 MiniMax-H3 B200 8-CTA FP4 量化（`fd9893c` `#7849`、`c4ea174` `#7862`）**：
  在 `cake_planner.py` 中为长 Pair-Mode 队列计划路由第二个 BF16 GQA 队列程序（`136.0 us -> 94.0 us`，**1.4468x**）；并为 `minimax_h3_varlen_attention` 引入 B200 尾块调度与 8-CTA FP4 量化器构建（`88.0 us -> 61.0 us`，**1.4426x**）。
- **行跨步 BF16 保持 Cluster-Split NVFP4 计划与 SM100 Split-K 提前单次读取 `alpha`（`ce6dc7d` `#7836`、`4c3d69a` `#7850`）**：
  允许 `cake_nvfp4_per_token` 在非连续行跨步 BF16 输入下继续复用 Cluster-Split 高速计划（`46.0 us -> 31.2 us`，**1.4744x**）；并在 `cake_blackwell_bf16_fp4` 的 SM100 小 M Split-K Reduce-Scatter 程序入口提前单次读取 `alpha`（`22.8 us -> 19.0 us`，**1.20x**）。

### 2.2 Triton Gluon/NVPTX TTI 与 TileLang SM120 块缩放代码生成
- **Triton Gluon 累加器释放与 NVPTX 保守 TTI 向量加载保留（`72bcaf2` `#12400`、`f37fda8` `#12337`）**：
  在 Gluon Blackwell 注意力与 Warp Specialization 算子中补齐 TMEM 累加器显式释放并拒绝不完整注意力 Tile（泄漏数 `0`）；在 `LLVMIRNVPTXTargetInfo.cpp` 中修正 `extractelement` 代价模型，防止 LLVM `VectorCombine` 错误重打包已解包至独立寄存器的向量加载。
- **TileLang SM120 块缩放 GEMM 奇数 Warp 网格、打包 FP8 向量拷贝与分区异步拷贝刷新（`7a5f446` `#3257`、`356f309` `#3276`、`134ade7` `#3278`）**：
  在 `gemm_blockscaled.cc` 中支持 SM120 块缩放 GEMM 片段与奇数 Warp Atom 网格；在 `cuda_fp8.h` 中保持 4 通道 FP8 向量打包拷贝；并在 `loop_partition.cc` 中刷新 Let 绑定以修复分区布局下的 `cp.async` 降级。

---

## 3. 离线算例验证与架构师量化结论

运行 `python experiments/flashinfer_vsa_sdsa_fp4_triton_gluon_tilelang_sm120_phase155/run.py --check` 得到的确定性验证指标如下：

| 评估维度 | 基线方案 | Phase 155 优化方案 | 量化收益 / 结果 |
|---|---|---|---|
| SDSA Token-Sparse MQA Decode (`#6420`) | 稠密 32K MQA `192.0 us` | Top-2048 词元稀疏解码 `28.0 us` | **6.8571x** 解码加速 |
| Hopper `cake_vsa_sm90` 长对队列 (`#7849`) | 单队列程序 `136.0 us` | 规划器路由第二 GQA 队列 `94.0 us` | **1.4468x** 吞吐提升 |
| MiniMax-H3 B200 变长注意力 (`#7862`) | 基线尾波次 `88.0 us` | B200 尾调度 + 8-CTA FP4 `61.0 us` | **1.4426x** 加速 |
| 行跨步 BF16 NVFP4 逐词元量化 (`#7836`) | 回退非分裂路径 `46.0 us` | 保持 Cluster-Split 计划 `31.2 us` | **1.4744x** 加速 |
| SM100 小 M Split-K Reduce-Scatter (`#7850`) | 循环重复读 `alpha` `22.8 us` | 入口单次早读 `alpha` `19.0 us` | **1.20x** 加速 |
