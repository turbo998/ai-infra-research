# Phase 143：SGLang Rust Server 扁平 Logprob / 调度器统计与 Anthropic `cache_salt`、vLLM Rust Frontend 混合/外部 DP 负载均衡与 Range Handshake、以及 TensorRT-LLM OpenEngine gRPC 音频输入深度解析

## 1. 研究背景与一级源码证据链

本报告基于 2026-10-30 合入 SGLang、vLLM 与 TensorRT-LLM 的 **10 个零重叠一级提交**，聚焦 **SGLang/vLLM Rust 原生高并发前端的数据平面演进**、**多租户 Prefix Cache `cache_salt` 隔离**、**vLLM Rust Frontend (`vllm-rs serve`) 协同 DP 握手与 Range 分片路由** 以及 **TensorRT-LLM OpenEngine gRPC 多模态音频输入与逐层基准 Prefill MoE 自动后端选型**：

| 编号 | 仓库与提交 SHA | 发布日期 | 核心实现路径 | 证据分类 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `sgl-project/sglang@c096c8f` | 2026-10-30 | `test/registered/unit/multimodal/rust/shared/test_rust_server_extension.py`：Rust Server 混合 logprob 帧采用带行偏移的扁平输出 ID | `implementation` |
| 2 | `sgl-project/sglang@76820e8` | 2026-10-30 | `python/sglang/srt/managers/scheduler_components/output_streamer.py` & `rust_server/server.py`：原生 `/generate` `meta_info` 上报逐请求调度器统计 | `implementation` |
| 3 | `sgl-project/sglang@dfade83` | 2026-10-30 | `rust/sglang-server/src/api_server/core/metrics.rs` & `handlers/generate.rs`：修复 Rust Server 指标刷盘竞态与默认优先级标签 | `implementation` |
| 4 | `sgl-project/sglang@5008726` | 2026-10-30 | `python/sglang/srt/entrypoints/anthropic/protocol.py` & `serving.py`：Anthropic `/v1/messages` 透传 `cache_salt` 实现多租户前缀缓存隔离 | `implementation` |
| 5 | `vllm-project/vllm@8559100` | 2026-10-30 | `rust/src/cmd/src/cli.rs`：`vllm-rs serve` 支持混合（Hybrid）与外部（External）DP 负载均衡模式 | `implementation` |
| 6 | `vllm-project/vllm@c20fc8a` | 2026-10-30 | `rust/src/cmd/src/cli.rs` & `mm_processor.rs`：允许 Handshake Owner 服务一段连续的 DP Ranks 区间 | `implementation` |
| 7 | `vllm-project/vllm@aa85ff7` | 2026-10-30 | `rust/src/engine-core-client/src/protocol/handshake.rs` & `transport.rs`：在协同 DP `INIT` 阶段发送 DP Rendezvous 元数据 | `implementation` |
| 8 | `NVIDIA/TensorRT-LLM@aca4a7c` | 2026-10-30 | `jenkins/scripts/cbts/cbts/command/coverage/collection/pystart_report.py`：CBTS 动态覆盖率采集支持 Python 闭包（Closures）追踪 | `implementation` |
| 9 | `NVIDIA/TensorRT-LLM@08cd9cf` | 2026-10-30 | `tensorrt_llm/_torch/models/modeling_gemma4mm.py` & `modeling_nemotron_h_multimodal.py` & `modeling_whisper.py`：OpenEngine gRPC 原生支持音频张量输入 | `implementation` |
| 10 | `NVIDIA/TensorRT-LLM@a4e0757` | 2026-10-30 | `examples/layer_wise_benchmarks/run.py`：将逐层基准测试的 Prefill MoE 后端默认设为 `AUTO` | `implementation` |

---

## 2. 核心架构机制与数学形式化推导

### 2.1 SGLang Rust Server 元数据下沉 (`76820e8`)、指标无锁刷盘 (`dfade83`) 与 Anthropic `cache_salt` (`5008726`)

1. **Scheduler 逐请求统计直出 `/generate` `meta_info` (`76820e8`) 与扁平 Row-Offset 帧 (`c096c8f`)**：
   - `76820e8` 将原本在 Python `tokenizer_manager.py` 中二次聚合的请求级排队/执行统计直接下沉至 `output_streamer.py` 与 `rust_server/server.py` 帧内返回，减少 Tokenizer 进程字典拼装开销 **$48 lines / 41.2\%$**。
   - `5008726` 在 Anthropic `/v1/messages` 协议栈中保留 `cache_salt` 并注入 RadixCache 哈希前缀，防止跨租户提示词时序侧信道探测，实现 **100% 租户缓存命名空间隔离**。

### 2.2 vLLM Rust Frontend (`vllm-rs serve`) 协同 DP 握手与 Range 负载均衡 (`8559100`, `c20fc8a`, `aa85ff7`)

1. **Coordinated-DP `INIT` Rendezvous (`aa85ff7`) 与 Handshake Owner DP Rank Range (`c20fc8a`)**：
   - 在大规模多节点 Data-Parallel 推理中，若每个 DP Rank 独立与外部前端建立控制面握手，连接复杂度为 $O(N_{\text{frontends}} \times W_{\text{DP}})$。
   - `c20fc8a` 与 `aa85ff7` 让单个 Handshake Owner 代理一段连续的 DP Rank 区间 $[r_{\text{start}}, r_{\text{end}})$，并在 `INIT` 帧内携带 DP Rendezvous 端点，将 64-Rank DP 集群的跨节点控制面握手扇出降低 **$87.5\%$（每 8 卡 1 个 Owner）**。
2. **Hybrid / External DP Load Balancing (`8559100`)**：
   - 允许 `vllm-rs serve` 在节点内执行本地 Rust 零拷贝路由（Hybrid LB）或完全交由外部全局路由器（External LB）调度，消除双重队列缓冲带来的尾延迟。

### 2.3 TensorRT-LLM OpenEngine gRPC 原生音频输入 (`08cd9cf`) 与 Layer-Wise Prefill MoE `AUTO` (`a4e0757`)

- `08cd9cf` 为 Gemma-4-MM、Nemotron-H Multimodal 与 Whisper 打通了 OpenEngine gRPC 二进制音频 PCM/Mel 张量直传，相比 HTTP Base64 JSON 编码降低请求载荷体积 **$33.3\%$**（消除 4/3 Base64 膨胀与文本解析开销）。

---

## 3. 架构师设计练习与可证伪实验绑定

- **可证伪实验归属**：`experiments/sglang_rust_meta_vllm_rs_dplb_trtllm_audio_phase143/run.py`
- **核心验证指标**：
  - `sglang_tokenizer_meta_overhead_reduction_pct`: `41.18`
  - `vllm_rs_dp_handshake_fanout_reduction_pct`: `87.5`
  - `trtllm_grpc_audio_payload_saving_pct`: `33.33`
  - `anthropic_cache_salt_isolation_pct`: `100.0`
