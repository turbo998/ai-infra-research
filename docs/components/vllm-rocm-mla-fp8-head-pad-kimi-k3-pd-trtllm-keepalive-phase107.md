# vLLM ROCm AITER MLA FP8 预填充头数填充门控、Kimi-K3 1P1D 主机内存预算与 TensorRT-LLM 静默流保活及轻量 MPI 停止分析 (Phase 107 · Priority 116)

> **English Version**: [vllm-rocm-mla-fp8-head-pad-kimi-k3-pd-trtllm-keepalive-phase107.en.md](vllm-rocm-mla-fp8-head-pad-kimi-k3-pd-trtllm-keepalive-phase107.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/vllm_rocm_mla_fp8_head_pad_kimi_k3_pd_trtllm_keepalive_phase107/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`vllm-project/vllm@d0c20bb` (`#61863`: `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`)**：
   - **ROCm AITER MLA FP8 预填充头数翻倍门控（`is_valid_fp8_prefill_num_heads`）**：ROCm 汇编级 FP8 MLA 预填充核要求每卡头数为 16 的倍数，非 16 倍数需通过 `repeat_interleave` 或 `tile+slice` 填充至 16 的倍数后再切片还原。当每卡头数为 `1..8`（例如 Kimi-K2.x 在 TP8 下为 `8` 头/卡）时，填充至 16 头会使有效计算量至少翻倍（`pad_multiplier = 2.0x`）并叠加张量复制/切片开销（相对代价 `11.2` vs 原生 BF16 预填充代价 `8.0`）。新增 `is_valid_fp8_prefill_num_heads` 对 `1..8` 头拒绝 FP8 预填充并保留无填充 BF16 预填充（在 8 头/卡下获得 **`1.40x` 预填充加速**），同时对 Kimi-K3 TP8（`12` 头/卡，填充至 16 头仅 `1.3333x`）与 16 对齐头数继续启用 FP8 预填充。
2. **`vllm-project/vllm@5f1a564` (`#63726`: `recipes/models/moonshotai/Kimi-K3.yaml`) & `7fe793a` (`#63769`)、`984a418` (`#63763`)**：
   - **AMD MI355X Kimi-K3 1P1D 分离式预填充/解码主机 DRAM 卸载预算刷新**：在双节点 8×MI355X（TP8/DCP8，MoRIIO READ 模式 + `SimpleCPUOffloadConnector`）Kimi-K3 + DSpark 投机解码配方中，将 8 卡聚合主机 DRAM KV 卸载预算 `CPU_OFFLOAD_BYTES` 从 `1,799 GB` 提升至 **`2,399 GB`（每 rank `299.875 GB`）**，使 64K 上下文并发缓存序列容量提升 **`1.3342x`（`393 -> 524` 条）**；同时 Rust Frontend 补齐 `--enable-log-outputs` / `--enable-log-deltas` 并保持 `top_logprobs` 引擎原生顺序。
3. **`NVIDIA/TensorRT-LLM@90e573e` (`#21061`: `tensorrt_llm/serve/responses_utils.py`) & `df95eda` (`#20434`: `tensorrt_llm/llmapi/_llmapi_mpi_stop.py`) & `b6751ca` (`#21097`)**：
   - **Responses API 静默流 `15.0s` 保活与无重型导入的独立 ZMQ MPI 停止器**：针对长排队、长上下文预填充或大模型生成完整工具调用（tool call）期间 SSE 流长时间静默被网关/代理（如 60s 阈值）断开的问题，`keep_responses_stream_alive` 每 **`15.0s`**（`RESPONSES_STREAM_KEEPALIVE_INTERVAL_S = 15.0`）重发一次 `event: response.in_progress` 帧（在 `95s` 静默期内发送 `6` 次保活帧，将网关断流率从 `100%` 降至 **`0%`**）；同时 `_llmapi_mpi_stop.py` 仅导入轻量 `pyzmq` 与标准库 `hmac`/`pickle` 发送认证停止信号，避免在停止辅助进程中冷导入整个 `tensorrt_llm` 与 MPI（启动耗时从 `14.5s` 降至 **`0.18s`，`80.56x` 加速**）。
4. **`sgl-project/sglang@4549937` (`#44487`)、`d3a157a` (`#46620`)、`f9d132b` (`#31286`)、`e4d4a9a` (`#46536`)**：
   - 修复 Nano-Nemotron-VL 嵌套 Puzzle 异构层配置丢失、清理 `CacheInitParams::cache_ttl_second` 冗余字段、移除启动路径 TorchScript 弃用调用并保护跨硬件平台的 CUDA RoPE 分支。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：设单头 BF16 预填充归一化代价为 `1.0`，16 头对齐的 FP8 ASM 预填充每头计算代价为 `0.62`，非对齐填充与切片附加代价为 `0.08 * padded_heads`。分别计算 `num_heads = 8`（Kimi-K2.x TP8）与 `num_heads = 12`（Kimi-K3 TP8）下填充至 `16` 头后的总代价（`11.2` vs `8.0` 及 `11.2` vs `12.0`），解释为何分界阈值恰好落在 `num_heads = 9`。
- **可证伪假设**：若在反向代理空闲超时 `60s` 的生产网关后运行 `95s` 工具调用生成而不发送 SSE 保活帧，流式连接将在首个工具调用 JSON 完成前被代理强制重置（`keep_responses_stream_alive` 通过 `15.0s` 周期重发 `response.in_progress` 将最大静默间隔封顶在 `15.0s` 内予以证伪）。
