# SGLang Paged-Experts 槽位查找融合与 PD 线程设备绑定、vLLM Rust gRPC 免反分词与推理强度映射及 TensorRT-LLM OpenEngine 全局传输 ID（Phase 102）

- **问题 ID**：`sglang-paged-experts-decide-pd-ctx-vllm-grpc-detok-trtllm-phase102`（`priority: 111`，组件：`inference`）
- **证据等级**：`own-offline`（基于上游源码补丁审计 + Python 3.12 标准库确定性算例 `experiments/sglang_paged_experts_pd_ctx_vllm_grpc_phase102/run.py`）
- **配套英文版**：[`sglang-paged-experts-decide-pd-ctx-vllm-grpc-detok-trtllm-phase102.en.md`](sglang-paged-experts-decide-pd-ctx-vllm-grpc-detok-trtllm-phase102.en.md)

## 1. 一手上游信源（零重叠）

1. **SGLang `8063296528` (#46457)**：`[Perf] paged-experts: fuse slot lookup into decide`
2. **SGLang `7fe5b127e4` (#45678)**：`[Fix] disaggregation: Avoid extra GPU contexts with PD`
3. **SGLang `a6f5bd397b` (#43815)**：`[Perf] [AMD] fp8: Use the AITER CK blockscale GEMM by default on gfx942`
4. **SGLang `f9257d6c7c` (#45576)**：`[Perf] [AMD] rccl: upgrade rccl to 2.32.3 for rocm10.1`
5. **SGLang `8a455f1b90` (#44888)**：`[Fix] diffusion: pipeline and model-helper bug fixes`
6. **vLLM `2fb5111bba` (#63789)**：`[Perf][Rust Frontend] Skip detokenization for gRPC requests that read no text`
7. **vLLM `803bc13cf9` (#62312)**：`[Rust Frontend] Add --reasoning-effort-map to rewrite request reasoning efforts`
8. **vLLM `4216a234cf` (#61877)**：`[Rust Frontend] Add per-request speculative decoding metrics`
9. **vLLM `447d28eef2` (#63292)**：`[Rust Frontend] Group prompt-cache usage and report created_cache_tokens`
10. **vLLM `40fe396b60` (#64032)**：`[Bugfix][Multimodal] Return 400 when local media access is disabled`
11. **TensorRT-LLM `2d3e60e27e` (#21043)**：`[None][fix] allocate global transfer IDs for OpenEngine prefill`
12. **TensorRT-LLM `7b12a0172e` (#21090)**：`[TRTLLM-17175][test] Add a CPU test for the CuTe DSL offline loader and fix two drivers`

---

## 2. 核心实现机制与定量分析

### 2.1 SGLang `paged_experts_decide` 融合槽位查找与权重写出 (`8063296528`)

在主机内存分页 MoE（Paged Experts）执行路径中，旧实现先运行 `paged_experts_decide` 与 `store.gather`，随后在 PyTorch 层执行 `r.expert_slot[topk_ids.clamp(min=0)].masked_fill(topk_ids < 0, -1)`，每层额外引入 4 个微型逐元素/索引算子启动。
- **融合机制**：`paged_experts_decide_kernel`（单 Warp LRU 决策核）在完成缺失专家槽位分配与 `src/dst/count` 换入计划写出的同时，直接将每个路由条目的槽位号与路由权重写入 `run_ids[T]` 与 `run_w[T]`；对越界/负值 Padding 条目（`id < 0` 或 `id >= E`）直接写入 `slot = 0, weight = 0.0f`。
- **定量收益**：每层 MoE 辅助启动次数从 **6 次降至 2 次**（削减 **66.67%**），在 61 层 DeepSeek-V3/R1 类模型上每步减少 **244 次** CUDA Kernel 启动开销。

### 2.2 SGLang PD 分离传输工作线程 GPU Context 隔离 (`7fe5b127e4`)

在 Mooncake 与 MoRI PD 分离后端中，新建 Python 线程默认绑定在 `device 0`。当 Rank `1..7` 的异步传输线程调用 `wait_event.synchronize()` 时，会在 `GPU 0` 上意外初始化 CUDA Primary Context。
- **修复机制**：新增 `bind_thread_to_device(gpu_id)` 与 `run_on_device(gpu_id, fn, *args)`，作为 `ThreadPoolExecutor(initializer=...)` 与后台线程入口。
- **显存收益**：在 8 卡节点上消除 `GPU 0` 上来自其余 7 个 Rank 的 **7 个冗余 CUDA 主上下文**，按每个上下文约 `420 MiB` 计，直接为 `GPU 0` 释放 **2,940.0 MiB** 可用 KV Cache 显存。

### 2.3 vLLM Rust 前端 gRPC 免反分词与 TensorRT-LLM OpenEngine 63-Bit 全局传输 ID (`2fb5111bba`, `803bc13cf9`, `2d3e60e27e`)

- **vLLM gRPC `output_text=false` 免反分词 (`2fb5111bba`)**：当下游仅消费 `output_token_ids`（如 RL 采样器或远程 PD 路由）并设置 `ResponseOptions.output_text = false` 时，Rust 前端将 `detokenize` 设为 `false`，在 `64 × 512 = 32,768` 输出 Token 批次上节省 **60.621 ms** 主机分词器 CPU 时间（调用次数降为 `0`）。
- **vLLM `--reasoning-effort-map` (`803bc13cf9`)**：仅改写类型化字段 `request.chat_options.reasoning_effort`（如 `"high" -> "max"`），而显式传入 `chat_template_kwargs["reasoning_effort"]` 的覆写值保持原样透传。
- **TensorRT-LLM OpenEngine 全局传输 ID (`2d3e60e27e`)**：多实例独立 Prefill 引擎生成的本地 `ctx_request_id`（如两个 Worker 同时分配 `274`）在汇聚至共享 Decode 池时会发生冲突。补丁为 `context_only` 请求分配 `MIN_GLOBAL_ID + secrets.randbelow((1 << 63) - MIN_GLOBAL_ID)`，将跨实例 ID 碰撞率降至 **0.0%**。

---

## 3. 架构师设计练习与可证伪问题

1. **练习计算**：若 8 卡节点每 Rank 启动 2 个 MoRI 传输队列但未绑定线程设备，`GPU 0` 会额外承受多少个进程级 Primary Context？（答案：CUDA Primary Context 按进程而非线程计数，因此其余 7 个进程各在 `GPU 0` 上打开 1 个 Primary Context，共 7 个）。
2. **可证伪假设**：在 vLLM gRPC 请求中若同时指定 `output_text = false` 且 `stop_strings` 非空，服务端必须拒绝该请求或要求显式文本解码，因为未反分词路径无法匹配字符串停止词。
