# vLLM `FusedMMInputNorm` 非持久缓冲、ROCm HY-V4 融合加速、Scale-Out `output_mode="derender"`、Rust 自定义直方图分桶与结构化输出优先级、TRT-LLM 视频哈希一致性与 SGLang NPU 初始化深度解析

> **语言 / Language**: [中文](vllm-fused-mm-norm-rocm-hyv4-scaleout-derender-trtllm-sglang.md) | [English](vllm-fused-mm-norm-rocm-hyv4-scaleout-derender-trtllm-sglang.en.md)  
> **Phase**: Phase 49 (`priority: 58`) · **日期**: 2026-10-22  
> **上游一手来源**:
> - `vllm-project/vllm@90c4a1a73eecea0428ee4303ad96c9fdeaac3f80` (`#62767`, 2026-10-21)
> - `vllm-project/vllm@833349105b9581d19487295454c19c924e8a3b37` (`#58509`, 2026-10-21)
> - `vllm-project/vllm@817fa3518cd4bcc94b31e044c534c002e701c801` (`#60082`, 2026-10-21)
> - `vllm-project/vllm@c1210ccbe5eab9ad5ee1213f0fc09c3ff13e4bea` (`#62826`, 2026-10-21)
> - `vllm-project/vllm@18e8791134e8e46134f982c8263d5fa0a50548c0` (`#62823`, 2026-10-21)
> - `NVIDIA/TensorRT-LLM@12accb2d5e888bf7dc003fc83c0993b4eab86a75` (`#19920`, 2026-10-21)
> - `sgl-project/sglang@2815eed49c94dbc3266fb5dc4b01c61fc8eb73ca` (`#43039`, 2026-10-21)

---

## 1. 背景与核心问题

在多模态预处理、ROCm HY-V4 模型适配、Scale-Out 解耦渲染/生成服务、vLLM Rust 前端以及 TensorRT-LLM / SGLang 跨硬件运行时中，上游近期解决了七个关键工程问题：

1. **`FusedMMInputNorm` 被 `DummyModelLoader` 覆盖问题 (`vllm#62767`)**：`FusedMMInputNorm` 将图像处理器（Image Processor）的 `rescale_factor / std` 与 `-mean / std` 注册为默认 `persistent=True` 的 PyTorch buffer。当使用 `--load-format dummy` 进行基准测试时，`DummyModelLoader` 会遍历 `state_dict()` 并将这两个非检查点（non-checkpoint）常量覆写为随机/常数填充值，导致视觉输入归一化完全失真。
2. **ROCm HY-V4 Preview 算子瓶颈与签名兼容 (`vllm#58509`)**：HY-V4 在 ROCm (`gfx942/gfx950`) 上缺乏融合的 IHC post/pre RMSNorm 核，AITER MXFP8 MoE 专家未放行 `MoEActivation.SILU`（仅放行了 `SILU_NO_MUL`），且 `deepgemm_fp8_paged_mqa_logits` 在旧版 aiter 下不支持 `ChunkedLocalAttentionMetadata` 新参数。
3. **Scale-Out `/v1/generate` 缺乏解耦 `derender` 模式 (`vllm#60082`)**：当无 GPU 的 `/render` 服务器将 Chat 请求预处理为 `GenerateRequest` 发往生成服务器时，生成端此前只能返回 `tokens` 或原始 `text`，无法在生成端直接调用工具调用解析器（tool-call parser）与推理内容解析器（reasoning parser）产出结构化 `ChatMessage`。
4. **vLLM Rust 前端监控与结构化输出对齐 (`vllm#62826`, `vllm#62823`)**：Rust 前端缺少 `--custom-histogram-buckets` 自定义 8 大指标族分桶能力，且在同时传入 `response_format` 与 `structured_outputs` 时会直接优先返回 `structured_outputs` 而忽略 `response_format` 覆盖规则。
5. **TensorRT-LLM 离线与在线视频缓存哈希不一致 (`TensorRT-LLM#19920`) & SGLang NPU 预热顺序 (`sglang#43039`)**：TRT-LLM 离线 `load_video` 走 `cv2` 临时文件而未计算 `VideoMediaIO` 的 `raw_bytes_hash`，导致离线与在线服务前缀缓存哈希不一致；SGLang 在 `init_parallel_runtime` 中未在 `_resolve_backend` 前调用 `init_npu_backend()`，导致 `--pre-warm-nccl` 在昇腾 NPU 上触发 `torch.cuda.*` 报错。

---

## 2. 核心架构与实现深挖

```
+---------------------------------------------------------------------------------------------------+
|               Inference Serving & Multimodal Runtime Upgrades (vLLM / TRT-LLM / SGLang)           |
+---------------------------------------------------------------------------------------------------+
| 1. FusedMMInputNorm (#62767)                 | 2. ROCm HY-V4 Preview Optimization (#58509)        |
|    - register_buffer("weight", ...,          |    - triton_ihc_post_pre_rms_norm (5 ops -> 1)     |
|      persistent=False)                       |    - AiterMxfp8Experts admits SILU + gate_mode     |
|    - Immune to DummyModelLoader overwrite    |    - _accepts_parameters for MQA logits            |
+----------------------------------------------+----------------------------------------------------+
| 3. Scale-Out output_mode="derender" (#60082) | 4. Rust Frontend Parity (#62826, #62823)           |
|    - /render emits ParseContext              |    - 8 BucketFamily custom histogram bounds        |
|    - Validated strictly only in derender     |    - response_format overrides same-kind constraint|
+----------------------------------------------+----------------------------------------------------+
| 5. TRT-LLM Video Cache Parity (#19920)       | 6. SGLang NPU Parallel Bootstrap (#43039)          |
|    - Offline load_video -> VideoMediaIO      |    - init_npu_backend() before _resolve_backend()  |
|    - Shared raw_bytes_hash + 200 MiB cap     |    - Routes torch.cuda.* -> torch.npu.* for warmup |
+----------------------------------------------+----------------------------------------------------+
```

### 2.1 `FusedMMInputNorm` 非持久缓冲 (`vllm#62767`)
- `FusedMMInputNorm.__init__` 预计算 `weight = rescale_factor / std` 与 `bias = -mean / std`（形状 `[1, C, 1, 1]`），并在 `register_buffer("weight", weight, persistent=False)` 与 `register_buffer("bias", bias, persistent=False)` 中显式设置 `persistent=False`。
- 由于非持久 buffer 不进入 `module.state_dict()`，模型权重加载器与 `DummyModelLoader` 均不会触碰它们，保证 dummy 压测与真实权重路径拥有完全一致的图像归一化数值（像素 `[128, 64, 192]` 下修复后误差 `< 1e-12`，而旧路径被覆写为 `0.5` 时最大绝对误差达 **`97.4946`**）。

### 2.2 ROCm HY-V4 Preview 算子融合与内省 (`vllm#58509`)
- **`triton_ihc_post_pre_rms_norm`**：在 `vllm/model_executor/models/hunyuan_v1.py` 中新增 `_ihc_post_pre_rms_norm_kernel`，将 5 个独立步骤——(1) `x` 的 FP32 均方根归一化并乘以 `w_post`、(2) 乘以流缩放因子 `scale_2` 并累加到多流状态 `x_h[B, n, C]`、(3) 原地写回更新后的 `x_h_next`、(4) 对 `n` 个流 masing-masing 计算 `w_pre` 的 RMSNorm、(5) 输出 `x_normed`——融合进**单个 Triton kernel**（每对 `(batch_idx, stream_idx)` 对应一个 program）。
- **`AiterMxfp8Experts` 门控激活对齐**：在 `rocm_aiter_fused_moe.py` 中新增 `fused_moe_supports_gate_mode()`，将 `MoEActivation.SILU` 与 `MoEActivation.GELU` 纳入 `AiterMxfp8Experts.supported_activations()`，并在 `activation == SILU_NO_MUL` 时向底层 `rocm_aiter_ops.fused_moe` 传递 `gate_mode=False`。
- **签名内省兼容**：通过 `inspect.signature` 的 `_accepts_parameters` 检查 `aiter.deepgemm_fp8_paged_mqa_logits` 是否支持 `chunked_Size` 与 `cu_seqlens_k`，实现新旧 ROCm AITER 版本的无感兼容。

### 2.3 Scale-Out `output_mode="derender"` 与 `ParseContext` (`vllm#60082`)
- **协议扩展**：`OutputMode` 新增 `"derender"`（与 `"tokens"`、`"text"` 并列）。`/render` 端点在 `render_chat_request` 中构造 `ParseContext`（包含 `tools`、`tool_choice`、`parallel_tool_calls`、`include_reasoning`、剔除 API-server 内部状态后的 JSON-safe `chat_template_kwargs` 以及 `history_tool_call_cnt`）。
- **分层校验契约**：`ParseContext` 配置 `extra="allow"`，使得不读取 `parse_context` 的中间层（如 `output_mode="tokens"` 的解耦 Prefill 节点）遇到新版 `/render` 发送的未知字段时不会拒绝请求；仅当最终生成节点指定 `output_mode="derender"` 时，`GenerateRequest` 才会严格校验并拒绝未知字段（如拼写错误的 `tool_choise`）或 `stream=True`，随后通过 `OnlineDerenderer._derender_text_choices` 执行工具调用与推理块解析。

### 2.4 Rust 前端、TensorRT-LLM 视频哈希与 SGLang NPU 初始化 (`#62826`, `#62823`, `#19920`, `#43039`)
- **Rust `--custom-histogram-buckets` (`vllm#62826`)**：在 `rust/src/metrics/src/buckets.rs` 中定义 8 个 `BucketFamily`（`request_latency`、`time_to_first_token`、`inter_token_latency`、`iteration_tokens`、`request_params_n`、`request_num_preemptions`、`request_tokens`、`kv_cache_residency`），反序列化时严格校验边界非空、有限、`> 0` 且严格单调递增，并在连接引擎前通过 `METRICS.set_custom_histogram_buckets` 锁定。
- **Rust `convert_from_response_format` (`vllm#62823`)**：非 `Text` 的 `response_format` 会覆盖 `structured_outputs` 中**同类型**（`std::mem::discriminant` 相同）的约束并保留其余选项，若两者约束类型不同则直接返回 `400 Invalid Request`。
- **TRT-LLM 离线视频哈希统一 (`TensorRT-LLM#19920`) & SGLang NPU 初始化 (`sglang#43039`)**：TRT-LLM 将离线 `load_video` 统一路由至 `VideoMediaIO`（共享 `_MAX_RESPONSE_BYTES = 200 MiB` 与 base64 长度上限 `279,620,268` 字符），确保 `path`、`file://`、`data:video/mp4;base64` 与 `http(s)://` 产生完全一致的 `raw_bytes_hash`；SGLang 在 `init_parallel_runtime` 入口处先执行 `if is_npu(): init_npu_backend()`，确保 `pre_warm_nccl` 正确重定向至 `torch.npu.*`。

---

## 3. 标准库确定性复现实验验证

通过 `experiments/vllm-fused-mm-norm-rocm-hyv4-scaleout-derender-trtllm-sglang/run.py --check` 验证：

| 验证维度 | 指标 / 不变量 | 实测 / 复现值 | 状态 |
|---|---|---|---|
| `FusedMMInputNorm` 数值保真 | `persistent=False` vs 被覆写 `0.5` 最大误差 | `< 1e-12` vs `97.4946` | `PASS` |
| ROCm HY-V4 IHC Norm 融合 | 每层 Kernel Launch 次数缩减 | `5 -> 1` (**`5.0×`** 发射缩减) | `PASS` |
| Scale-Out `ParseContext` 分层校验 | Prefill (`tokens`) 放行未知字段 / `derender` 拒绝 | `ok:tokens` / `error:unknown_parse_context_fields:tool_choise` | `PASS` |
| Rust 前端分桶与结构化输出 | 8 大指标族单调递增校验 & 同类约束覆盖 | `8` families validated, `json_schema:schema_v2` | `PASS` |
| TRT-LLM 视频哈希与 `200 MiB` 上限 | 4 种输入源 `raw_bytes_hash` 一致性 | `trtllm_video_hash_parity = True` (`209,715,200 B` cap) | `PASS` |
