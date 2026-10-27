# TransformerEngine DeepSeek-V4 自定义 MoE 路由、Megatron-LM 多模态等待队列批量编码、TorchTitan Qwen3 RoPE 校准与 Mooncake 多缓冲区 / Dynamo DeepStack 预算深度解析（Phase 104 · Priority 113）

> **English Version**: [te-moe-dsv4-routing-megatron-vlm-torchtitan-mooncake-lmcache-dynamo-phase104.en.md](te-moe-dsv4-routing-megatron-vlm-torchtitan-mooncake-lmcache-dynamo-phase104.en.md)
> **实验脚本**: [`experiments/te_moe_dsv4_routing_megatron_vlm_mooncake_batch_phase104/run.py`](../../experiments/te_moe_dsv4_routing_megatron_vlm_mooncake_batch_phase104/run.py) · **定量汇总**: [`summary.json`](../../experiments/te_moe_dsv4_routing_megatron_vlm_mooncake_batch_phase104/summary.json)
> **7-Card 视觉报告**: [`docs/reports/ai-infra-phase104-benchmark-cards.html`](../reports/ai-infra-phase104-benchmark-cards.html)

---

## 1. 证据分级与上游源码锚点（Evidence Classification）

| 证据 ID / 来源 | 仓库与 Commit | 关键路径 | 证据分级 |
|---|---|---|---|
| `src-te-moe-dsv4-custom-routing-expert-counts-2026` | `NVIDIA/TransformerEngine@83d8c4e3370bc0117189adbbd88a50fd57440acd` | `tests/jax/test_te_ep_moe.py` | `implementation` |
| `src-megatron-inference-batch-waiting-image-encodes-2026` | `NVIDIA/Megatron-LM@1142209a19054a89dba0c868d6af2cb474152bd9` | `megatron/core/inference/engines/dynamic_engine.py` | `implementation` |
| `src-torchtitan-qwen3-235b-a22b-rope-theta-1e6-2026` | `pytorch/torchtitan@13806cbae05bb9a6ceecc0ce9aa0642d53e7ba25` | `torchtitan/models/qwen3/flavors.py` | `implementation` |
| `src-torchtitan-attn-gym-defer-lazy-backend-error-2026` | `pytorch/torchtitan@3ed5aa976fdf38e05fd50885ab192ea66ed6c5b7` | `torchtitan/models/common/attn_gym.py` | `implementation` |
| `src-mooncake-http-metadata-bearer-auth-bounds-2026` | `kvcache-ai/Mooncake@f7baeb138db8f14bb158e9fc6c84d1a1bd6178f6` | `mooncake-store/tests/e2e/run_nof_heartbeat_tcp_e2e.sh` | `implementation` |
| `src-mooncake-store-c-rust-multi-buffer-batch-io-2026` | `kvcache-ai/Mooncake@679ae2517249d38ca99f18ea640b3524cb6dd180` | `mooncake-store/include/store_c.h`, `mooncake-store/rust/src/ffi_dlopen.rs` | `implementation` |
| `src-mooncake-efa-discover-via-libfabric-2026` | `kvcache-ai/Mooncake@b03155412e7d41782b24558548b952fd7f2d6ee3` | `mooncake-transfer-engine/src/transport/efa_transport/efa_context.cpp` | `implementation` |
| `src-lmcache-sm80-bf16-fp8-key-store-upcast-2026` | `LMCache/LMCache@f5feba8054e82ce56d531661379ab1d7d87b316f` | `lmcache/v1/distributed/serde/turboquant/store_kernel.py` | `implementation` |
| `src-dynamo-vllm-deepstack-multimodal-cache-budget-2026` | `ai-dynamo/dynamo@2e96c98ba75372091b6f2bdc1f5b08e13f5a5dc5` | `components/src/dynamo/vllm/multimodal_utils/multimodal_embedding_cache_connector.py` | `implementation` |
| `src-nixl-gpunetio-reject-failed-oob-ipv4-early-2026` | `ai-dynamo/nixl@42f1714f46054e4ef9b16605a51bba03255c278a` | `src/plugins/gpunetio/gpunetio_backend.cpp` | `implementation` |

---

## 2. 核心实现路径、数据布局与同步机制

### 2.1 TransformerEngine DeepSeek-V4 自定义 MoE 路由与 Megatron-LM 视觉批量预编码 (`83d8c4e337`, `1142209a19`)
- **TE `RoutingMapInfo` 自定义路由 (`83d8c4e337`)**：为支持 DeepSeek-V4 的外部自定义/哈希路由，TE 将路由契约拆分为 `RouterComputationInfo`（TE 内部计算门控 GEMM）与 `RoutingMapInfo`（调用方直接传入 `routing_map` 与 `routing_scales`），并同时返回 `expert_counts` 供负载均衡损失使用。在 $T=4096, H=7168, E=256$（BF16）下，跳过冗余 TE 内部门控 GEMM 每层可直接节省 **`15.03 GFLOPs`** 与 **`61.50 MiB` HBM 读写流量**。
- **Megatron-LM 多模态等待队列批量图像编码 (`1142209a19`)**：`DynamicInferenceEngine` 在 prefill 调度前扫描等待队列，通过 `_has_deferred_image_encodes` 快速旁路纯文本请求（开销 `0.0 us`），并将 `8` 个等待中的多模态图像合并为 `1` 次批量 ViT 前向，将视觉编码耗时从 `42.4 ms`（`8` 次串行启动）压降至 `14.5 ms`（**`2.9241x` 加速**）。

### 2.2 TorchTitan `Qwen3-235B-A22B` RoPE 基频修正与 `attn_gym` 惰性后端容错 (`13806cbae0`, `3ed5aa976f`)
- **RoPE 基频修正 (`13806cbae0`)**：`Qwen3-235B-A22B` 官方发布配置为 `rope_theta = 1,000,000.0`（`1e6`），而 TorchTitan 误填为 `5,000,000.0`（`5e6`，仅 `Qwen3-235B-A22B-Instruct-2507` 256K 变体使用），消除了加载开源权重时的 **`5.0x` 波长畸变**。
- **`attn_gym` 惰性导入 (`3ed5aa976f`)**：在未安装 Hoppper/Blackwell 专属 `cutedsl` 的环境中，仅在实际调用对应后端时抛出错误，保证常规 FlexAttention / SDPA 训练正常启动。

### 2.3 Mooncake 零拷贝多缓冲区批处理、HTTP 鉴权、EFA `fi_getinfo()` 与 LMCache / Dynamo / NIXL (`f7baeb138d`, `679ae25172`, `b03155412e`, `f5feba8054`, `2e96c98ba7`, `42f1714f46`)
- **Mooncake C ABI / Rust 多缓冲区批处理 (`679ae25172`)**：新增 `mooncake_store_batch_put_from_multi_buffers` 与 `batch_get_into_multi_buffers`，允许将 `61` 层非连续 KV Cache 切片（每层 `64 KiB`）在 **单次（`61 -> 1`，`61.0x` RPC 缩减）** 零拷贝调用中完成传输，每块节省 **`3,997,696 B`（`3.8125 MiB`）** 主机/设备暂存拷贝；同时为 HTTP metadata server 增加 Bearer token 鉴权与容量上限 (`f7baeb138d`)，并通过 `fi_getinfo()` 发现 AWS EFA 设备以彻底移除 `<infiniband/verbs.h>` 头文件耦合 (`b03155412e`)。
- **LMCache Ampere SM80 FP8 存储修复 (`f5feba8054`)**：在 `_tq_fused_store_fp8` 中先将 `tl.bfloat16` 升精度为 `tl.float32` 再转 `tl.float8e4b15`，修复 A100 上的 Triton 编译崩溃。
- **Dynamo vLLM DeepStack 多模态缓存预算校准 (`2e96c98ba7`)**：对于 Qwen3-VL 等 3 层 DeepStack 视觉编码器，单行特征宽度为 $3 \times H = 12,288$（`24,576 B`）而非 $H = 4,096$（`8,192 B`）。通过 `mm_encoder_output_widths` 实测宽度计费，消除了 **`3.0x`（低估 `66.67%`）** 的显存/主机缓存预算漏算。
- **NIXL GPUNetIO 早期 OOB IPv4 校验 (`42f1714f46`)**：在 `gpunetio_backend` 初始化阶段立即拒绝失败的 OOB IPv4 网卡发现，避免延迟到數據传输阶段触发静默超时。

---

## 3. 架构师设计练习与后续可证伪问题

- **设计练习**：若某 VLM 的 DeepStack 层数为 $L_{\text{ds}} = 4$，主干 `hidden_size = 5120`（BF16），每张图像产生 `1024` 个视觉 tokens，求单张图像在 Dynamo 多模态缓存中的真实字节占用：
  $$\text{Bytes} = 1024 \times (4 \times 5120) \times 2\text{ B} = 41,943,040\text{ B} = 40\text{ MiB}$$
- **可证伪问题**：
  1. 在 Mooncake 61 层 `batch_put_from_multi_buffers` 中，当单层切片小于 `4 KiB` 时，RDMA scatter-gather SGE 链表长度上限（如 ConnectX-7 的 `max_sge`）是否会触发分批回退？
