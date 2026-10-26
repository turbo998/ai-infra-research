# SGLang 流水线 Ulysses 注意力、DSV4 TBO 对齐、AMD gfx950 KDA 融合解码、HiCache L3 键布局、Confirmed-First 路由器、vLLM DSV4.1 DSpark PCP、OOT 块缩放 FP8、H100 批不变矩阵乘、KV Offload 前缀保序与 TensorRT-LLM Mamba 缓存模块化深挖 (Phase 95 / Priority 104)

> **证据分类 (Evidence Class)**：上游源码实现 (`implementation`) + 本项目 Python 3.12 标准库离线确定性模型 (`own-offline`)。

## 1. 核心架构问题与一手源码落点

| 信源 ID | 仓库与提交 | 关键实现路径与架构不变量 |
|---|---|---|
| `sglang-diffusion-ulysses-pipelined-own-row-copy-pr46308-2026` | `sgl-project/sglang` @ `9841183827` (PR #46308) | 多模态扩散模型（WanVideo 等）流水线 Ulysses 序列并行注意力将 `fill_all(blocks)` 重构为 `fill_all(blocks, part)`：先对 `q` (`part=0`) 执行 RMSNorm+RoPE 并发起 IPC All-to-All 交换，同时在后台隐藏 `k` (`part=1`) 的本卡行归一化与拷贝（模型化隐藏 `42.0 us`，准备总耗时从 `274.0 us` 降至 `232.0 us`）。 |
| `sglang-dsv4-non-ep-tbo-attn-tp-fix-pr33250-2026` | `sgl-project/sglang` @ `ef11eb3b76` (PR #33250) | 修复 DeepSeek-V4 在非 EP 且 `attn_tp > 1` 下的 Two-Batch Overlap (TBO) 对齐错误：新增 `_pad_tbo_child_inputs`，令 decode 子批次按实际 decode token 总数（如 `16 reqs * 2 tokens = 32`）而非 `batch_size` (16) 进行填充，消除 `prepare_mlp_sync_batch` 跨 TP 秩形状失配。 |
| `sglang-amd-gfx950-kda-kimi-k3-hip-fused-decode-pr45713-2026` | `sgl-project/sglang` @ `6a28c96a49` (PR #45713) | AMD MI350X (`gfx950`) 默认启用 Kimi-K3 KDA HIP 融合 decode 核函数（`kda_decode_gfx950.cu`）：每个 `(token, head)` 分配一个 256 线程 workgroup（每秩 12 个 128 维 head），单核融合 `f_b` 投影、depthwise `conv1d + SiLU`、Q/K L2 归一化 + `128x128` FP32 状态 delta-rule 更新及输出门控 RMSNorm（4 次启动合并为 1 次，且保护 Mamba 池 `slot 0` padding 槽位）。 |
| `sglang-hicache-mooncake-l3-object-key-layout-pr40467-2026` | `sgl-project/sglang` @ `64fc5f0bdb` (PR #40467) | HiCache 通过 `merge_scheduler_info_layouts` 聚合各 PP 阶段与 DP 秩的 Mooncake L3 内存池描述符并向外部 KV 感知路由器暴露精确对象键格式（`{key_prefix}...`），使前端路由器可直接查询 L3 缓存命中率。 |
| `sglang-router-confirmed-first-pending-prefix-cap-pr46505-2026` | `sgl-project/sglang` @ `eabb11a4c1` (PR #46505) | `sgl-router` 新增 `--cache-pending-prefix-consult-below <ratio>` 与 follower cap：仅当已确认（confirmed）KV 事件前缀覆盖率低于 `ratio` 且在途（pending）前缀严格更深时才参考 pending 质押，并限制跟风并发数，防止冷启动突发流量压垮单台 worker。 |
| `vllm-dsv41-dspark-pcp-spec-decode-pr61994-2026` | `vllm-project/vllm` @ `af6665b5dd` (PR #61994) | vLLM 为 DeepSeek-V4.1 支持 Prefill Context Parallelism (`pcp_size` $\in \{2, 4\}$) 与定长 DSpark 投机解码（`tail_size=1`），并在配置校验阶段显式拦截不兼容的 MTP 与自适应（adaptive）DSpark。 |
| `vllm-oot-block-scaled-fp8-linear-registration-pr62026-2026` | `vllm-project/vllm` @ `5be249d0ad` (PR #62026) | `register_linear_kernel` 扩展 Out-of-Tree (OOT) 线性核注册类型，从原有的 `mp`/`int8`/`fp8` 3 类扩展至 `fp8_block`/`mxfp8`/`nvfp4`/`mxfp4`/`mxfp6` 共 8 类块缩放低精度核。 |
| `vllm-h100-batch-invariant-matmul-llama31-qwen3-pr62219-2026` | `vllm-project/vllm` @ `c84ff6b3f4` (PR #62219) | 为 H100 批不变（batch-invariant）确定性矩阵乘表新增 Llama-3.1-8B 与 Qwen3-8B（如 `gate_up_proj` `(24576, 4096)`，`block_k=128`，覆盖 `M=1..2048` 10 个分桶）调优配置。 |
| `vllm-kv-offload-preserve-running-lru-prefixes-pr61050-2026` | `vllm-project/vllm` @ `e113018f50` (PR #61050) | 修复 CPU KV Offload 在批量写入时破坏前缀连续性的缺陷：`CachePolicy.insert(self, key, chunk, req_context)` 通过 `req_context.get_offload_key_position(key)` 获取各块在序列中的位置，确保容量受限时保留最前面的连续前缀块（如 `0..3`），避免因 `chunk 0` 先被淘汰导致整条前缀失效。 |
| `trtllm-modular-mamba-cache-managers-pr18843-2026` | `NVIDIA/TensorRT-LLM` @ `ba6c1fc87d` (PR #18843) | 模块化 Mamba 缓存管理器：在 `_bind_step_buffers` 中一次性绑定每层 `mamba_layer_cache` 视图（48 层 × 100 步下将视图构造从 `9,600` 次降至 `96` 次），并强制 `_advance_replay_state` 按实际接受的 draft token 数 + bonus token 推进状态。 |

## 2. 可复现实验

- **实验命令**：`python experiments/sglang_ulysses_dsv4_tbo_gfx950_kda_hicache_router_vllm_dspark_fp8_trtllm_mamba_phase95/run.py --check`
