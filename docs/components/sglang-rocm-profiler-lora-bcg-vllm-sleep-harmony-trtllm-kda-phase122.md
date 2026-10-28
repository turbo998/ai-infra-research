# SGLang ROCm 10.1 Profiler 环境变量清理、Diffusion LoRA 与可中断 CUDA Graph 预分配缓冲、vLLM Sleep 状态遥测与 Harmony 推理 Token 计量、TensorRT-LLM CuteDSL NVFP4 MoE 与 KDA Prefill 深度解析（Phase 122）

> **English Version**: [sglang-rocm-profiler-lora-bcg-vllm-sleep-harmony-trtllm-kda-phase122.en.md](sglang-rocm-profiler-lora-bcg-vllm-sleep-harmony-trtllm-kda-phase122.en.md)
> **所属阶段**: Phase 122 (`priority: 131`) · Track B 推理运行时、性能剖析、遥测与混合注意力
> **实验脚本**: [`experiments/sglang_rocm_profiler_lora_bcg_vllm_sleep_harmony_trtllm_kda_phase122/run.py`](../../experiments/sglang_rocm_profiler_lora_bcg_vllm_sleep_harmony_trtllm_kda_phase122/run.py)

---

## 1. 一手来源与证据分类（Primary Evidence）

| 证据 ID / 来源 | 仓库与 PR/Commit | 合入日期 | 证据分级 | 核心贡献 |
|---|---|---|---|---|
| `src-sglang-rocm-101-clear-rocprofiler-register-library-env-pr44292-2026` | [`sgl-project/sglang#44292`](https://github.com/sgl-project/sglang/pull/44292) (`291f38c3`) | 2026-10-28 | `implementation` + `author-benchmark` | 在导入 `torch` 前读取 `torch/version.py` 并清理子进程继承的 `ROCPROFILER_REGISTER_LIBRARY`，恢复 ROCm 10.1 / PyTorch 2.12 (`gfx1250`) 上的 GPU 轨迹采集（`kernel: 0 -> 5,331`，`cuda_runtime: 0 -> 9,533`，`ac2g: 0 -> 14,864`，总计恢复 **29,728** 个 GPU/关联事件） |
| `src-sglang-diffusion-lora-move-alpha-rslora-float-alpha-pr45833-2026` | [`sgl-project/sglang#45833`](https://github.com/sgl-project/sglang/pull/45833) (`67b3c9f9`) | 2026-10-27 | `implementation` | 修复 Diffusion LoRA 的 3 项静默缩放失真：在 Flux 拆分投影间传递 `move_alpha`、用 `.removesuffix(".weight")` 兼容无后缀 Kohya 键以正确施加 RSLoRA $\sqrt{r}$ 缩放、用 `float()` 替代 `int()` 保留小数 `alpha`（如 `0.5`） |
| `src-sglang-diffusion-breakable-cuda-graph-caller-allocated-buffers-pr46660-2026` | [`sgl-project/sglang#46660`](https://github.com/sgl-project/sglang/pull/46660) (`a912002e`) | 2026-10-27 | `implementation` | 将 4 处 Diffusion 可中断 CUDA Graph 断点（`MiniMaxH3` 注意力、`VDNH3` 混合注意力、`MiniMaxH3DiTModel._embed`、`Qwen-Image 2.1` `attend_past_prefix_pad`）重构为调用方预分配输出缓冲区并强制校验 `shape/dtype` |
| `src-vllm-rust-frontend-engine-sleep-state-prometheus-metrics-pr63959-2026` | [`vllm-project/vllm#63959`](https://github.com/vllm-project/vllm/pull/63959) (`9df11cf6`) | 2026-10-28 | `implementation` | vLLM Rust 前端新增 `vllm:engine_sleep_state{model_name, engine, sleep_state}`（`awake`、`weights_offloaded`、`discard_all`）与强类型 `SleepLevel` (`0, 1, 2`) 校验 |
| `src-vllm-rust-frontend-gpt-oss-harmony-reasoning-token-count-pr63960-2026` | [`vllm-project/vllm#63960`](https://github.com/vllm-project/vllm/pull/63960) (`00d4b782`) | 2026-10-28 | `implementation` | 在 vLLM Rust OpenAI 前端中精确计量 `gpt-oss` Harmony 流式与非流式 `reasoning_tokens`（仅统计 `analysis` 与含工具 `recipient` 的 `commentary` 正文 token，排除控制 token 与 `final`/无接收方前导消息） |
| `src-trtllm-cutedsl-nvfp4-moe-reenable-large-intermediate-fp32-chain-pr21101-2026` | [`NVIDIA/TensorRT-LLM#21101`](https://github.com/NVIDIA/TensorRT-LLM/pull/21101) (`f6c4125c`) | 2026-10-28 | `implementation` + `author-benchmark` | 重新启用 `intermediate_size >= 14336`（Mixtral `14336`、Grok `32768`）的 CuteDSL NVFP4 MoE 测试，证明 CuteDSL 与 TRTLLM-Gen 均为 **0% 失配**，而与参考实现的 `0.9%–3.4%` 差异源于参考实现在 `fp4_quantize` 前的双重 BF16 舍入（相对精确 FP32 链在旧阈值下产生 `9.8%–27.9%` 偏差） |
| `src-trtllm-flashinfer-kda-prefill-fused-cutedsl-conv-silu-pr20653-2026` | [`NVIDIA/TensorRT-LLM#20653`](https://github.com/NVIDIA/TensorRT-LLM/pull/20653) (`222676c8`) | 2026-10-23 | `implementation` | 集成 FlashInfer `BT16` / `Small-BH` KDA prefill 与单内核 CuTe DSL 融合短卷积 + SiLU + 原地状态池更新（每线程 `8` 通道 `16B`、3-token 寄存器历史、`h < -2` warp-vote 精确指数回退），B200 上 **293 项单测全过** |

---

## 2. 核心架构机制与公式推导

### 2.1 SGLang ROCm 10.1 Profiler 修复 (`#44292`) 与 Diffusion LoRA / BCG 缓冲 (`#45833`, `#46660`)

1. **ROCm 10.1 `ROCPROFILER_REGISTER_LIBRARY` 继承死锁修复 (`#44292`)**：在 ROCm 10.1 + PyTorch 2.12 中，主进程导入 `torch` 时 HIP/HSA 初始化会将 `ROCPROFILER_REGISTER_LIBRARY=librocprofiler-sdk-roctx.so` 写入环境变量并被 `spawn` 子进程继承。当子进程初始化时，`rocprofiler-register` 看到该变量即跳过内部表构建并试图把运行时符号转发给尚未加载的 `rocprofiler-sdk`，导致 `torch.profiler` 丢失全部 GPU 事件。通过在导入 `torch` 前直接解析文件系统中的 `torch/version.py` 并清除未配对的 `ROCPROFILER_REGISTER_LIBRARY`，`gfx1250` 上 FLUX.2-dev 的 GPU 轨迹从 `0` 恢复至 **29,728** 个事件（`5,331` kernel + `9,533` `cuda_runtime` + `14,864` `ac2g`）。
2. **Diffusion LoRA 小数 Alpha 与 RSLoRA 缩放 (`#45833`)**：修复了 `int(alpha)` 将 `alpha = 0.5` 截断为 `0` 的致命退化，并使用 `.removesuffix(".weight")` 确保无后缀 Kohya H3 键同样乘以 RSLoRA 因子 $\alpha / \sqrt{r}$（当 $r=4$ 时为标准 $\alpha / r$ 的 **2.0x**）。

### 2.2 vLLM Sleep 状态遥测 (`#63959`) 与 Harmony Reasoning Token 计量 (`#63960`)

- **Harmony `reasoning_tokens` 边界公式 (`#63960`)**：
  $$N_{\text{reasoning}} = \sum_{m \in \mathcal{M}} \mathbb{I}\big[\text{channel}(m) = \texttt{analysis} \;\lor\; (\text{channel}(m) = \texttt{commentary} \land \text{recipient}(m) \neq \varnothing)\big] \cdot |T_{\text{body}}(m)|$$
  所有 `<|channel|>`、`<|message|>`、`<|end|>`、`<|call|>` 控制 token 与消息头均被严格排除；当多字节 UTF-8 字符被切分到两个 chunk 时，前半段 token 计入下一个产生非空文本的 delta。

### 2.3 TensorRT-LLM CuteDSL NVFP4 MoE 精度归因 (`#21101`) 与 KDA Fused Conv+SiLU (`#20653`)

- **NVFP4 MoE 双重 BF16 舍入归因 (`#21101`)**：CuteDSL 与 TRTLLM-Gen 在 FC1 后保持 FP32 累加器直通 `routing_scale -> SwiGLU -> fp4_quantize`（中间仅 1 次 BF16 转换），两者在 `intermediate_size = 14336 / 32768` 上 **0% 失配**；参考实现执行了两次独立 BF16 舍入，导致在 NVFP4 量化的离散跳变边界上偏离精确 FP32 链达 `9.8%–27.9%`（在 `atol=0.1, rtol=0.05` 下 CuteDSL vs Ref 失配仅 `0.9%–3.4% < 7%`）。
- **KDA Prefill CuTe DSL 短卷积 SiLU 快速近似误差界 (`#20653`)**：令 $h = x / 2$，快速近似 $\text{silu}(x) \approx h + h \cdot \texttt{tanh.approx}(h)$ 的相对误差上界为：
  $$\varepsilon_{\text{rel}}(h) \le 2^{-11} \cdot \frac{|\tanh(h)|}{1 + \tanh(h)}$$
  当 $h \ge -2$ 时 $\varepsilon_{\text{rel}} \le 1.30\%$；当 $h \to -\infty$ 时因 $1 + \tanh(h) \to 0$ 发生相消放大。内核在每 8-token 组上做一次 warp 投票，一旦任意 lane 出现 $\min(h) < -2$，即在循环后用保次正规数的 $x \cdot e / (1 + e), \; e = \exp(-|x|)$ 覆写修正。
