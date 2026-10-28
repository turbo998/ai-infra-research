# SGLang ROCm 10.1 Profiler Inheritance Fix, Diffusion LoRA & Breakable CUDA Graph Buffers, vLLM Sleep State Telemetry & Harmony Reasoning Token Accounting, and TensorRT-LLM CuteDSL NVFP4 MoE & KDA Prefill (Phase 122)

> **中文版**: [sglang-rocm-profiler-lora-bcg-vllm-sleep-harmony-trtllm-kda-phase122.md](sglang-rocm-profiler-lora-bcg-vllm-sleep-harmony-trtllm-kda-phase122.md)
> **Phase**: Phase 122 (`priority: 131`) · Track B Inference Runtime, Profiler, Telemetry & Hybrid Attention
> **Experiment Script**: [`experiments/sglang_rocm_profiler_lora_bcg_vllm_sleep_harmony_trtllm_kda_phase122/run.py`](../../experiments/sglang_rocm_profiler_lora_bcg_vllm_sleep_harmony_trtllm_kda_phase122/run.py)

---

## 1. Primary Sources & Evidence Classification

| Evidence / Source ID | Repo & PR/Commit | Merge Date | Classification | Core Technical Contribution |
|---|---|---|---|---|
| `src-sglang-rocm-101-clear-rocprofiler-register-library-env-pr44292-2026` | [`sgl-project/sglang#44292`](https://github.com/sgl-project/sglang/pull/44292) (`291f38c3`) | 2026-10-28 | `implementation` + `author-benchmark` | Reads `torch/version.py` before importing `torch` and clears inherited `ROCPROFILER_REGISTER_LIBRARY` on ROCm 10.1 / PyTorch 2.12+, restoring **29,728** GPU/correlation trace events (`5,331` kernel, `9,533` `cuda_runtime`, `14,864` `ac2g`) on `gfx1250` |
| `src-sglang-diffusion-lora-move-alpha-rslora-float-alpha-pr45833-2026` | [`sgl-project/sglang#45833`](https://github.com/sgl-project/sglang/pull/45833) (`67b3c9f9`) | 2026-10-27 | `implementation` | Fixes 3 silent diffusion LoRA bugs: propagates `move_alpha` across split Flux projections, uses `.removesuffix(".weight")` so Kohya H3 keys receive RSLoRA $\sqrt{r}$ scaling, and uses `float()` instead of `int()` to preserve fractional `alpha` (`0.5`) |
| `src-sglang-diffusion-breakable-cuda-graph-caller-allocated-buffers-pr46660-2026` | [`sgl-project/sglang#46660`](https://github.com/sgl-project/sglang/pull/46660) (`a912002e`) | 2026-10-27 | `implementation` | Refactors 4 diffusion breakable CUDA graph sites (`MiniMaxH3` attention, `VDNH3` hybrid attention, `MiniMaxH3DiTModel._embed`, `Qwen-Image 2.1` `attend_past_prefix_pad`) to write into caller-allocated output buffers with strict `shape/dtype` assertions |
| `src-vllm-rust-frontend-engine-sleep-state-prometheus-metrics-pr63959-2026` | [`vllm-project/vllm#63959`](https://github.com/vllm-project/vllm/pull/63959) (`9df11cf6`) | 2026-10-28 | `implementation` | Exports `vllm:engine_sleep_state{model_name, engine, sleep_state}` (`awake`, `weights_offloaded`, `discard_all`) and validates `SleepLevel` (`0, 1, 2`) in the vLLM Rust frontend |
| `src-vllm-rust-frontend-gpt-oss-harmony-reasoning-token-count-pr63960-2026` | [`vllm-project/vllm#63960`](https://github.com/vllm-project/vllm/pull/63960) (`00d4b782`) | 2026-10-28 | `implementation` | Counts `gpt-oss` Harmony `reasoning_tokens` in the Rust OpenAI frontend across `analysis` and tool-targeted `commentary` messages while excluding control tokens, headers, and `final`/preamble messages |
| `src-trtllm-cutedsl-nvfp4-moe-reenable-large-intermediate-fp32-chain-pr21101-2026` | [`NVIDIA/TensorRT-LLM#21101`](https://github.com/NVIDIA/TensorRT-LLM/pull/21101) (`f6c4125c`) | 2026-10-28 | `implementation` + `author-benchmark` | Re-enables CuteDSL NVFP4 MoE tests for `intermediate_size >= 14336` (Mixtral `14336`, Grok `32768`), proving **0% mismatch** between CuteDSL and TRTLLM-Gen FP32 chains vs reference double-BF16 rounding (`0.9%–3.4% < 7%` threshold) |
| `src-trtllm-flashinfer-kda-prefill-fused-cutedsl-conv-silu-pr20653-2026` | [`NVIDIA/TensorRT-LLM#20653`](https://github.com/NVIDIA/TensorRT-LLM/pull/20653) (`222676c8`) | 2026-10-23 | `implementation` | Integrates FlashInfer `BT16` / `Small-BH` KDA prefill with a single CuTe DSL fused short-conv + SiLU + indexed state update kernel (`8` channels/thread, 3-token register history, warp-vote fallback when `h < -2`; **293 passed** on B200) |

---

## 2. Core Architectural Analysis & Derivations

### 2.1 SGLang ROCm 10.1 Profiler Fix (`#44292`) & Diffusion LoRA / BCG Buffers (`#45833`, `#46660`)

1. **ROCm 10.1 `ROCPROFILER_REGISTER_LIBRARY` Inheritance Fix (`#44292`)**: On ROCm 10.1 + PyTorch 2.12, importing `torch` in the parent process sets `ROCPROFILER_REGISTER_LIBRARY=librocprofiler-sdk-roctx.so`, which spawned worker processes inherit before `rocprofiler-sdk` is loaded. `rocprofiler-register` then skips internal registration table creation and drops all GPU activity records. Inspecting `torch/version.py` directly on disk and clearing inherited `ROCPROFILER_REGISTER_LIBRARY` before `import torch` restores **29,728** GPU/correlation trace events (`5,331` kernel + `9,533` `cuda_runtime` + `14,864` `ac2g`) on `gfx1250` FLUX.2-dev.
2. **Diffusion LoRA Fractional Alpha & RSLoRA Scaling (`#45833`)**: Replaces `int(alpha)` (which truncated `alpha=0.5` to `0`) with `float(alpha)` and uses `.removesuffix(".weight")` so suffix-free Kohya H3 keys receive RSLoRA $\alpha / \sqrt{r}$ scaling (**2.0x** at $r=4$ vs $\alpha / r$).

### 2.2 vLLM Sleep State Telemetry (`#63959`) & Harmony Reasoning Token Accounting (`#63960`)

- **Harmony `reasoning_tokens` Accounting Rule (`#63960`)**:
  $$N_{\text{reasoning}} = \sum_{m \in \mathcal{M}} \mathbb{I}\big[\text{channel}(m) = \texttt{analysis} \;\lor\; (\text{channel}(m) = \texttt{commentary} \land \text{recipient}(m) \neq \varnothing)\big] \cdot |T_{\text{body}}(m)|$$
  Control tokens (`<|channel|>`, `<|message|>`, `<|end|>`, `<|call|>`) and message headers are excluded, and split multi-byte UTF-8 tokens are buffered until the next non-empty text delta.

### 2.3 TensorRT-LLM CuteDSL NVFP4 MoE Precision (`#21101`) & KDA Fused Conv+SiLU (`#20653`)

- **NVFP4 MoE Double-BF16 Rounding Attribution (`#21101`)**: Both CuteDSL and TRTLLM-Gen keep FC1 accumulators in FP32 through `routing_scale -> SwiGLU -> fp4_quantize` (**0% mismatch** between the two kernels at `intermediate_size = 14336 / 32768`), whereas the reference path rounds to BF16 twice before quantization (`9.8%–27.9%` drift against an exact FP32 chain under old tolerances, vs `0.9%–3.4% < 7%` under `atol=0.1, rtol=0.05`).
- **KDA Prefill CuTe DSL SiLU Fast-Approximation Bound (`#20653`)**: With $h = x / 2$, $\text{silu}(x) \approx h + h \cdot \texttt{tanh.approx}(h)$ satisfies:
  $$\varepsilon_{\text{rel}}(h) \le 2^{-11} \cdot \frac{|\tanh(h)|}{1 + \tanh(h)} \le 1.30\% \quad (h \ge -2)$$
  A single warp vote per 8-token group flags groups where any lane has $h < -2$ and recomputes them via $x \cdot e / (1 + e), \; e = \exp(-|x|)$ to prevent cancellation as $h \to -\infty$.
