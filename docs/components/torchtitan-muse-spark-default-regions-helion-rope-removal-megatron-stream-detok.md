# TorchTitan Muse Spark 默认局部编译区域下沉、Helion RoPE 覆盖移除与非连续 `ComplexRoPE` 验证及 Megatron-LM 流式增量反分词器对齐

**English Companion**: [torchtitan-muse-spark-default-regions-helion-rope-removal-megatron-stream-detok.en.md](torchtitan-muse-spark-default-regions-helion-rope-removal-megatron-stream-detok.en.md)

## 1. 背景与架构问题

在大规模 MoE 训练与推理引擎（TorchTitan Muse Spark 816B-A42B、DeepSeek-V3 MTP、Megatron-LM 动态推理服务）中，编译器区域配置、外部算子依赖与流式文本后处理存在三类典型工程缺陷：
1. **Muse Spark 局部编译区域仅限 Scale 配方的遗漏**：`MuseSparkModel` 声明了 `muse_spark_input_scaled_rmsnorm`、`muse_spark_logit_postprocess` 与 `muse_spark_residual` 三个融合区域，但此前仅在 Dist-MoE Scale 配方中通过 `extend` 启用，Base、Debug 与 RL DAPO（含 Generator CUDA Graph）配方仍停留在默认的 `["loss", "fp32_to_bf16_split", "rope"]`。在 4k Token 下，Eager 模式的单层 `residual`（`724 us` vs 带宽理想值 `95 us`）、`post-FFN norm`（`381 us` vs `47 us`）与 `expert-output norm`（`1335 us` vs `189 us`）合计耗时 **`2,440 us`**（带宽有效利用率仅 **`13.57%`**）。
2. **CUDA 专属 `helion_rope.py` 维护负担与 ROCm 阻塞**：随着编译版 `ComplexRoPE` 在 128k 上下文以内全面追平或超越 Helion，`torchtitan_recipes/overrides/helion_rope.py`（`1,031` 行代码 + `helion==1.1.0` 依赖）已无任何生产配方调用，却导致 `deepseek_v3_mtp_fsdp+ep` 集成测试在 ROCm 上被强制标记 `skip_rocm_test=True`。
3. **Megatron-LM 流式与非流式输出不一致**：`HuggingFaceFastIncrementalDetokenizer` 使用 `DecodeStream(skip_special_tokens=True)` 时，会将 Tokenizer 中所有标记为 `special=True` 的 `AddedToken`（包括聊天/推理模型必需的 `<think>`、`</think>`、`<tool_call>`、`</tool_call>`）全部剥离，而最终非流式响应仅剥离 `all_special_ids` 与末尾 EOS，导致流式拼接文本丢失思维链与工具调用标签。

## 2. 一手源码实现路径（Implementation Audit）

### 2.1 `pytorch/torchtitan` (`39bd2054d9c4fcbcabc343d262abad85f19020fb`, `#5433`, `2026-10-20`)
- **`torchtitan/models/muse_spark/model.py` & `torchtitan_recipes/models/muse_spark.py`**：
  - 将 `MuseSparkModel.Config.local_compile_regions` 默认列表直接扩展为 `["loss", "fp32_to_bf16_split", "rope", "muse_spark_input_scaled_rmsnorm", "muse_spark_logit_postprocess", "muse_spark_residual"]`，并删除 Scale 配方中冗余的 `_MUSE_SPARK_SCALE_LOCAL_COMPILE_REGIONS` 扩展（LoRA SFT 配方仍显式覆盖为 `["loss"]`）。
  - **作者实测（`author-benchmark`，GB300 2 层 816B-A42B EP32）**：
    - 相比 PR 1：训练（2k/4k/16k）延迟下降 **`-6.9%` 至 `-10.0%`**（相比 `main` 下降 **`-13.6%` 至 `-16.6%`**，如 4k CUDA Graph 从 `42.13 -> 39.28 -> 35.94 ms`）；
    - 预填充（2k/16k）下降 **`-10.0%` 至 `-10.9%`**（相比 `main` **`-14.6%` 至 `-15.7%`**）；
    - 解码（1/8/64/256 序列）下降 **`-7.1%` 至 `-11.2%`**（256 序列从 `1.55 -> 1.49 -> 1.32 ms`，相比 `main` **`-14.6%`**）；
    - 全训练+推理形状下 `residual` 仅生成 **`4` 张图**、`input_scaled_rmsnorm` 生成 **`6` 张图**（均低于 `recompile_limit=8`）。

### 2.2 `pytorch/torchtitan` (`abdf14320a01212430b4ddfb2d147a4dfbab4b3e`, `#5530`, `2026-10-20`)
- **`tests/unit_tests/gpu/models/qwen3_5/test_rope_compile.py` & `tests/integration_tests/models.py`**：
  - 彻底移除 `helion_rope.py`（576 行）及其测试（455 行）与 `helion==1.1.0`，并在 `deepseek_v3_mtp_fsdp+ep` 中解除 `skip_rocm_test=True`；
  - 新增 `test_compiled_matches_eager_complex_non_contiguous`：验证转置头步长的 `query = query_leaf.transpose(0, 1)`（步长 `(64, 4096, 1)`）与切片后插入单维的 `key = key_leaf[..., -64:].unsqueeze(1)`（步长 `(96, 64, 1)`）在不触发连续 `int32` 打包捷径时，编译路径与 Eager 路径前反向梯度完全一致。

### 2.3 `NVIDIA/Megatron-LM` (`ebab8e2e21c77361fd9edbaee3e3d7d90748f340`, `#8268`, `2026-10-20`)
- **`megatron/core/inference/text_generation_server/dynamic_text_gen_server/incremental_detokenizer.py`**：
  - 用精确集合 `self._skipped_token_ids = frozenset(huggingface_tokenizer.all_special_ids) | model_eos_token_ids(tokenizer)` 替代 `self._skip_special_tokens = True`，使流式增量解码保留 `<think>`、`</think>`、`<tool_call>`、`</tool_call>` 等结构化标记。

## 3. 确定性实验复算（`own-offline`）

运行 `python experiments\torchtitan_muse_default_regions_rope_megatron_detok\run.py --check`：

| 子系统 | 关键核验指标 | 修复前 / 基线 | 修复后 (`Phase 37`) | 证据分级 |
|---|---|---|---|---|
| Muse Spark 默认编译区域 | 4k 单层 3 算子 Eager vs 理想带宽 | `2,440.0 us` (`13.57%` 效率) | **默认启用 6 区域（理想上限 `331.0 us`）** | `author-benchmark` + `own-offline` |
| Muse Spark GB300 2L 延迟 | 4k 训练 Graph / 256 序列解码 | `42.13 ms` / `1.55 ms` (`main`) | **`35.94 ms` (`-14.7%`) / `1.32 ms` (`-14.6%`)** | `author-benchmark` |
| TorchTitan ComplexRoPE | 非连续步长 `(64,4096,1)` & `(96,64,1)` | 依赖 `helion==1.1.0` (`skip_rocm`) | **删 `1,031` 行、解封 ROCm、`0.0` 编译误差** | `implementation` + `own-offline` |
| Megatron 流式反分词器 | `<think>plan</think>answer<tool_call>call</tool_call>` | `"plananswercall"`（标签丢失） | **100% 匹配非流式输出并剥离 EOS** | `implementation` + `own-offline` |
