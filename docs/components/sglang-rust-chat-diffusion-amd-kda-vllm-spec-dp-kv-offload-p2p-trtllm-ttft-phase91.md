# Phase 91 Track B：SGLang Rust Chat 处理器与扩散图像原子发布/GLM 像素保真、AMD gfx950 KDA 分块预填充、vLLM 投机解码 DP 副本对齐与 P2P KV 卸载加固、以及 TensorRT-LLM 64M 元素 KV 扫描与 TTFT 分位数 GEN 准入控制

> **English Version**: [sglang-rust-chat-diffusion-amd-kda-vllm-spec-dp-kv-offload-p2p-trtllm-ttft-phase91.en.md](sglang-rust-chat-diffusion-amd-kda-vllm-spec-dp-kv-offload-p2p-trtllm-ttft-phase91.en.md)  
> **阶段状态**：Phase 91（`priority: 100 — 100-Priority Milestone!`）· 关联问题 `q-sglang-rust-chat-diffusion-amd-kda-vllm-spec-dp-kv-offload-p2p-trtllm-ttft-phase91` · 关联论断 `claim-sglang-rust-chat-diffusion-amd-kda-vllm-spec-dp-kv-offload-p2p-trtllm-ttft-phase91`  
> **可复现实验**：[`experiments/sglang_rust_chat_diffusion_amd_kda_vllm_spec_dp_kv_offload_p2p_trtllm_ttft_phase91/`](../../experiments/sglang_rust_chat_diffusion_amd_kda_vllm_spec_dp_kv_offload_p2p_trtllm_ttft_phase91/)

---

## 1. 一手源码证据矩阵（10 项上游提交）

| # | 仓库与提交 | 日期 | 证据级别 | 核心架构机制 |
|---|---|---|---|---|
| 1 | `sgl-project/sglang` [`911693e3f8`](https://github.com/sgl-project/sglang/commit/911693e3f8cc5d8dbdd5490934f448cb7db9722f) (`#45449`) | 2026-10-26 | `implementation` | `rust-server` 统一通过 `sglang-processor` 执行 Chat 模板渲染、分词与服务，移除 `dynamo-parsers` / `dynamo-renderer` / `minijinja` 等 6 个冗余 Rust 依赖并透传 `default_chat_template_kwargs` |
| 2 | `sgl-project/sglang` [`fe4ccb37e4`](https://github.com/sgl-project/sglang/commit/fe4ccb37e400ef857816a18f383f965661139ff4) (`#46060`) | 2026-10-26 | `implementation` | `sglang.multimodal_gen` 新增 `save_image_bytes_to_path`，通过同目录临时文件 + `os.replace` 原子发布输入图像，防止写入异常截断已有文件 |
| 3 | `sgl-project/sglang` [`d3e84c32e5`](https://github.com/sgl-project/sglang/commit/d3e84c32e5858c553463686f5f9471d68702efbe) (`#46193`) | 2026-10-26 | `implementation` | 扩散分离式推理（disaggregated diffusion）在动态批次切分时为 head 角色完整保留每个请求的 GLM 原始像素数据 |
| 4 | `sgl-project/sglang` [`c8398cb3a3`](https://github.com/sgl-project/sglang/commit/c8398cb3a378d62451023482ca3d20713aa59b78) (`#45710`) | 2026-10-26 | `implementation` | AMD `gfx950`（MI350X / MI355X）引入原生 HIP 分块预填充内核 `kda_prefill_gfx950.cu`（`+2898` 行），加速 Kimi Delta Attention |
| 5 | `vllm-project/vllm` [`29be2f2560`](https://github.com/vllm-project/vllm/commit/29be2f2560f0a4bb86796340928f8b2d56ab2d98) (`#62301`) | 2026-10-26 | `implementation` | 独立投机解码草稿模型（standalone draft model）在构造 `draft_vllm_config` 时显式继承主模型 `parallel_config.data_parallel_index` |
| 6 | `vllm-project/vllm` [`a64fc94081`](https://github.com/vllm-project/vllm/commit/a64fc94081494957dbdc0c3605b4b1ab2867bc0e) (`#60574`) | 2026-10-26 | `implementation` | KV Offload 调度器在单步内发生 prefix cache reset 触发抢占并立即重新加载时，保留新排队加载任务而不被抢占通知误清空 |
| 7 | `vllm-project/vllm` [`f682f1dc32`](https://github.com/vllm-project/vllm/commit/f682f1dc322bbbc69c390209e15fb90910f9d4c0) (`#63664`) | 2026-10-26 | `implementation` | P2P KV 分层卸载会话在 `_poll_registration` 发送 `ConnectAckMsg` 时改调 `self._do_send(...)`，发送异常立即将死会话标记为 `dead` |
| 8 | `vllm-project/vllm` [`71294c2e9a`](https://github.com/vllm-project/vllm/commit/71294c2e9ac41ed710b6eb4ca41cba3793705345) (`#63498`) | 2026-10-26 | `implementation` | P2P NIXL 传输代理在创建 UCX 后端时传入 `engine_config="TLS=^rc_gda"`，在纯主机 DRAM 分层通道中排除 GPU-direct `rc_gda` 车道 |
| 9 | `NVIDIA/TensorRT-LLM` [`c3c1ad3ab8`](https://github.com/NVIDIA/TensorRT-LLM/commit/c3c1ad3ab84c811a5c2841a946c841dd3c8c2b1c) (`#20928`) | 2026-10-26 | `implementation` | Warmup 后 KV Cache NaN/Inf 扫描改为每次扫描 `_INVALID_VALUE_SCAN_ELEMS = 1 << 26`（`67,108,864` 元素，较 256 页切片减少 `16x` 内核启动）并缓存 FP8 `isinf` 不支持状态 |
| 10 | `NVIDIA/TensorRT-LLM` [`dd75334918`](https://github.com/NVIDIA/TensorRT-LLM/commit/dd75334918500a9c0ad662b156afc9b49aeb234b) (`#20736`) | 2026-10-26 | `implementation` | 分离式推理新增按编排器初始到达时间 `orig_elapsed_s` 排序的 `GenAdmissionWaitingQueue`（`TLLM_GEN_ADMISSION_TTFT_PERCENTILE`），并修复 SWA scratch 复用下跳过前向后的精确 chunk 重放 |

---

## 2. 关键实现路径与架构深挖

### 2.1 vLLM P2P KV 卸载死会话收敛与 `TLS=^rc_gda` 隔离（`#60574`, `#63664`, `#63498`）
在多机多卡 P2P 主机内存 KV 分层卸载中：
1. **单步重置重载保护（`#60574`）**：当 `reset_prefix_cache()` 在同一调度步内使某请求被抢占（进入 `preempted_req_ids`）并立即重新排队触发新的异步 KV 卸载加载时，旧逻辑在 `_build_connector_meta` 尾部无条件清空 `preempted_req_ids` 的所有传输任务，导致新启动的重载任务丢失。修复后仅清理抢占前遗留的旧任务，保留同步新入队的重载任务。
2. **NIXL UCX `TLS=^rc_gda`（`#63498`）**：P2P KV 分层仅在主机固定内存（pinned DRAM）视图上运行；若 UCX 默认启用 GPU-direct 异步队列（`rc_gda`），对端阻塞时会触发 GPU 侧连接超时。显式配置 `engine_config="TLS=^rc_gda"` 彻底隔离 GPU Direct 车道。

### 2.2 TensorRT-LLM 64M 元素 KV 扫描与分离式推理 TTFT 分位数准入（`#20928`, `#20736`）
- **KV NaN/Inf 扫描加速（`#20928`）**：将 256 页小切片合并为每次最多 `67,108,864`（`1 << 26`）个连续元素的扁平视图调用，并在首次捕获 `torch.isinf` 对 FP8 抛出异常后缓存该能力位，将 20 亿元素缓冲区的扫描调用次数从 `512` 次降至 `32` 次（减少 `16x`）。
- **TTFT 分位数 GEN 准入与 SWA 重放（`#20736`）**：在 Context/Gen 分离式推理中，长提示词请求在 Context 节点耗时更长，若 Gen 等待队列仅按入队时间 FCFS 排序，长请求到达 Gen 后仍要排在短请求之后，导致尾部 TTFT 恶化。`GenAdmissionWaitingQueue` 基于全局单调时钟差 `orig_elapsed_s` 优先准入最早到达编排器的请求。同时在 SWA（滑动窗口注意力）scratch 页面复用下，若某次 chunked context forward 被跳过，`_pregrown_context_tokens` 强制下次重放保持已扩容的完整 chunk 长度，防止滑窗内出现未分配页面索引。
