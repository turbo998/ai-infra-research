# SGLang MNNVL NVML 织网检测、SM107 KDA Spec-Verify BV=4、DCP Block-FP8 Q-Proj 复制、vLLM FP8 Indexer WK 持久化与 TensorRT-LLM Gemma4 / INT8-WOQ MoE 对齐

> English version: [sglang-mnnvl-kda-sm107-dcp-qproj-vllm-wk-cublas-nixl-trtllm-gemma4-int8-woq.en.md](sglang-mnnvl-kda-sm107-dcp-qproj-vllm-wk-cublas-nixl-trtllm-gemma4-int8-woq.en.md)
> 证据等级：`implementation` + `author-benchmark`（上游源码/PR 实测）+ `own-offline`（本项目 Python 3.12 标准库确定性仿真器，非本项目 GPU 硬件实测）

## 1. 核心架构问题与源码演进路径

Phase 84（`priority: 93`）针对推理引擎栈（`sgl-project/sglang`、`vllm-project/vllm`、`NVIDIA/TensorRT-LLM`）的 9 个最新提交进行了源码级审计：

1. **SGLang MNNVL NVML 织网注册检测与 SM107 KDA Spec-Verify `BV=4` 窄分块（[#46463](https://github.com/sgl-project/sglang/commit/39b54ae0eeca90f7097b22fe016e9212523a5968)、[#46465](https://github.com/sgl-project/sglang/commit/640e11ebc0dfc660ea766acf43b0586d1819f494)）**：
   - `is_mnnvl_fabric_device()` 过去仅通过设备名称匹配 `GB200`/`GB300`，导致上报通用设备名的 `SM107` 多节点 NVLink 系统无法启用 NCCL cuMem/MNNVL、Custom AllReduce v2 multinode 与 DCP `fi_a2a`。新实现补充了通过 NVML 查询 NVLink fabric 注册状态的检测路径。
   - 在 `SM107`（212 SMs）上执行 Kimi-K3 TP8 的 KDA 投机验证（`target_verify`）递推核时，默认 `BV=32` 单 Warp 配置仅启动 `12 heads × 4 = 48` 个 CTA，严重欠利用 212 个 SM。将块宽缩窄至 `BV=4` 后，独立更新的 V 列被展开为 **384 个 CTA（8 倍并发）**，隔离核延迟从 `13.66 us` 降至 `6.99 us`（**1.954x** 加速）且输出完全一致。
2. **SGLang DCP `128x128` Block-FP8 `q_b_proj` 复制与 DeepSeek-V4 MoE 局部完成守卫（[#46359](https://github.com/sgl-project/sglang/commit/f264183bfe1d6f7e8f9046d5234f22e0496ecac7)、[#46365](https://github.com/sgl-project/sglang/commit/ad950cdcd9666a104321deeaf1cbf883b4e8c60b)）**：
   - `--dcp-replicate-q-proj` 在初始化阶段一次性 All-Gather 各 Rank 的 `q_b_proj`/`w_kc` 分片，从而在每层 MLA Decode 中消除一次 Query All-Gather 集合通信。本提交将该能力从仅限 BF16/FP16 扩展至 **128×128 Block-FP8 `q_b_proj`**（同步收集 FP8 权重与块缩放因子并按本地分片后处理）。
   - DeepSeek-V4 `prepare_output()` 修复了 `self.rows.group is None` 时的提前完成漏洞：仅当 `post_experts_output_is_complete(is_tp_path=False)` 确认 A2A Combine 已完成专家求和时才标记 `summed=True`，防止未求和回退路径静默跳过规约。
3. **vLLM DeepSeek-V3.2 FP8 `indexer.wk` 跨 `load_weights` 保持、cuBLAS Workspace 释放与 NIXL OBJ 遥测（[#59223](https://github.com/vllm-project/vllm/commit/2c880f1e68200f79c4663dd0f1e7beab6f536cfa)、[#63451](https://github.com/vllm-project/vllm/commit/b3466364b69db84ddeca30435f56c822551f8b14)、[#55576](https://github.com/vllm-project/vllm/commit/4a81a5097631ad9851a7045ebb4f2f8c4873eec0)）**：
   - 乱序权重流（如 `runai_streamer`）中 `lm_head.weight` 会触发 `AutoWeightsLoader` 的 `model` / 非 `model` 前缀切换并将加载拆分为两次 `load_weights()` 调用。若 `indexer.wk` 权重与其 `weight_scale_inv` 被拆在两次调用中，旧临时缓冲区会丢失其中一半，导致融合 `wk_weights_proj` 保留未初始化的随机值；修复后跨调用持久化挂起对。
   - `WeightCacheDaemon` 在预热后除释放通信组外，显式调用 `torch._C._cuda_clearCublasWorkspaces()` 释放缓存的 cuBLAS GEMM Scratch（TP8 下可回收约 `256 MiB` 驻留显存），仅保留 CUDA IPC 权重。
   - KV Offload 的 `ObjAgent` 启用 `nixl_agent_config(backends=[], capture_telemetry=True)`，修复 `get_xfer_telemetry()` 因默认关闭遥测而抛出 `RuntimeError` 的问题。
4. **TensorRT-LLM Gemma4 多模态编码器预算调度与 INT8-WOQ MoE 64 行对齐填充（[#20663](https://github.com/NVIDIA/TensorRT-LLM/commit/abb94ab8fc1c93ee377a5137c9b36466e2ae3664)、[#16198](https://github.com/NVIDIA/TensorRT-LLM/commit/64ef54ce1249f89ea23b03d4868425c3d5aec268)）**：
   - `Gemma4mm` 在多模态编码器预算下按项调度视觉/音频 Token，而无独立视觉塔的 `Gemma4Unified` 显式关闭编码器内存剖析与调度（返回空预算字典）。
   - `INT8-woq` MoE 混合 GEMM 预处理器要求行分块为 `rows_per_tile = 128 * 8 // 16 = 64` 的整数倍。当 `moe_intermediate_size = 1856`（`64 × 29`）按 `TP=2/4/8` 切分时，每分区维度（`928/464/232`）不再被 64 整除；新逻辑将其向上对齐至 64 倍数（`960/512/256`）并零填充，消除了 `TP>1` 权重加载崩溃。

## 2. 可复现实验与架构师练习

- 运行命令：`python experiments/sglang_mnnvl_kda_sm107_dcp_qproj_vllm_wk_cublas_nixl_trtllm_gemma4_int8_woq/run.py`
- 关联证据 ID：`ev-sglang-mnnvl-kda-sm107-dcp-qproj-vllm-wk-cublas-nixl-trtllm-gemma4-int8-woq-2026`
- **架构师设计练习**：验证当 `moe_intermediate_size = 1856` 且采用门控激活（SwiGLU，`expand_ratio = 2`）在 `TP=4` 下切分时，单分区未对齐维度 `464` 向上填充至 `512` 后的 FC1 扩展维度 `expand_padded = 1024` 带来的额外权重存储比例（`(512 - 464) / 464 = 10.34%`）。
