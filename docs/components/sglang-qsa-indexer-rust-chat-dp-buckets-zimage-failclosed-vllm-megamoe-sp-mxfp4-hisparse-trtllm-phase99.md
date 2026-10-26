# SGLang QSA 长缓存前缀 Indexer 优化、Rust Chat 协议对齐、DP-Attention 图桶与空闲 Rank 跳过、Z-Image RoPE 隔离与 Diffusion 故障闭合，vLLM SP 分片 MegaMoE 对称缓冲、Kimi-K3 融合 KV 打包、MXFP4 3x 激活量化与 HiSparse 哈希探测缩减及 TensorRT-LLM 宽束搜索 Host 减负（Phase 99 · Priority 108）

> **English Edition**: [sglang-qsa-indexer-rust-chat-dp-buckets-zimage-failclosed-vllm-megamoe-sp-mxfp4-hisparse-trtllm-phase99.en.md](sglang-qsa-indexer-rust-chat-dp-buckets-zimage-failclosed-vllm-megamoe-sp-mxfp4-hisparse-trtllm-phase99.en.md)

## 1. 背景与核心问题

在长上下文稀疏注意力（QSA/HiSparse）、多 Rank DP-Attention 推理、多模态 Diffusion 服务以及 Blackwell/Hopper MoE 量化（MXFP4/FP8）中，推理引擎面临五大系统性损耗：

1. **SGLang QSA 长缓存前缀 Indexer 宽度膨胀（`sglang#45791`, `642e9c1448`）**：在命中长 Prefix Cache 的 Prefill 阶段，QSA Indexer 此前按全局最大块表宽度分配和扫描 logits，而非按主机已知的请求实际最大压缩块数 `prefill_max_row_keys` 裁剪。
2. **SGLang Rust Chat 层一致性、DP-Attention 图桶对齐与 Diffusion 隔离/故障闭合（`sglang#45659/#46138/#46044/#46025`）**：
   - `#45659` (`4fb422b9fe`) 使 `rust/sglang-processor` 在聊天模板、`continues_final_message`、流式输出和 lowered-chat 错误码上与 Python 层完全一致；
   - `#46138` (`9b230c1dae`) 在 DP-Attention 下对齐各 DP Rank 的 Prefill CUDA Graph 桶与 Decode Padding，并新增 `skips_prefill_metadata_on_idle_dp_rank` 使空闲 DP Rank 重放 Padded Prefill 图时无需构造冗余注意力元数据；
   - `#46044` (`17204f3482`) 修复 Z-Image Diffusion 跨请求复用旧 RoPE 表导致的图像/文本错位（按 stream 维护 `_rope_caches`）；
   - `#46025` (`0f41fffec8`) 在本地 Worker 丢失或多 Rank OOM 后使 `/health` 立即返回 `503`（Fail-Closed），防止把流量路由给半死进程。
3. **vLLM DeepGEMM MegaMoE SP 对称缓冲、Kimi-K3 融合 KV 打包、MXFP4 3x 激活量化与 HiSparse 探测优化（`vllm#63541/#61527/#63101/#61263`）**：
   - `#63541` (`2feedbe8dc`) 将 DeepGEMM MegaMoE 的对称通信缓冲区按单个序列并行（SP）分片大小 `max_num_batched_tokens // sp_size` 分配，在 `SP=8` 下减少 `87.5%`（`8.0x`）显存占用；
   - `#61527` (`4eb5df8038`) 将 Kimi-K3 FP8 分块上下文 K/V 打包路由至 FlashInfer `fused_mla_kv_pack_quant_fp8` 融合核；
   - `#63101` (`f3bdb0e025`) 优化 GLM-5.3 等模型的 MXFP4 专家激活量化核（`mxfp4_experts_quant` / `silu_and_mul_mxfp4_experts_quant`），算子提速 **`3.0x`**、端到端吞吐提升 **`+5.0%`**；
   - `#61263` (`a2ecc8ac4d`) 降低 HiSparse 稀疏 MLA 驻留哈希表的负载因子（`<= 1/4`），显著减少哈希冲突探测次数。
4. **TensorRT-LLM SID-GR 宽束搜索每步 Host 开销消减（`tensorrt-llm#20830`, `832b2cb7f4`）**：在 `beam_width=64` 的宽束搜索（Wide Beam Search）中精简 Decoder 输入打包与 Sampler 主机侧 Python 循环开销。

## 2. 定量建模与架构验证（`own-offline`）

通过确定性标准库模型 [`experiments/sglang_qsa_indexer_rust_chat_dp_buckets_zimage_failclosed_vllm_megamoe_sp_mxfp4_hisparse_trtllm_phase99/run.py`](../../experiments/sglang_qsa_indexer_rust_chat_dp_buckets_zimage_failclosed_vllm_megamoe_sp_mxfp4_hisparse_trtllm_phase99/run.py) 验证：

| 引擎与优化路径 | 基线表现 | Phase 99 优化后表现 | 关键量化收益 / 架构不变量 |
| --- | --- | --- | --- |
| SGLang QSA 长前缀 Prefill Indexer (`#45791`) | 全局 2,048 块宽度（`3.84 ms`） | 请求级 256 块宽度（`0.62 ms`） | 宽度缩减 **`8.0x`**，Indexer 加速 **`6.194x`** |
| SGLang DP-Attention 图桶对齐 + Diffusion 保护 (`#46138/#46044/#46025`) | 7 次跨 Rank 桶失配 / 旧 RoPE 串扰 | 统一桶对齐 + 按流 RoPE + Fail-Closed | **`0`** 桶失配、**`0`** RoPE 串扰、**`0`** 僵尸 `200 OK` |
| vLLM DeepGEMM MegaMoE SP 对称缓冲 (`#63541`, `SP=8`) | `224.0 MiB`（全量 `max_num_batched_tokens`） | `28.0 MiB`（单 SP 分片容量） | 对称缓冲显存缩减 **`8.0x`**（`-87.5%`） |
| vLLM Kimi-K3 FlashInfer 融合 FP8 KV 打包 (`#61527`) | 分离 Concat + Quant（`1.44 ms`） | 融合 `fused_mla_kv_pack_quant_fp8`（`0.52 ms`） | **`2.769x`** KV 打包加速 |
| vLLM MXFP4 MoE 激活量化核 (`#63101`) | `114.0 us` | `38.0 us` | **`3.0x`** 算子加速，**`+5.0%`** 端到端吞吐提升 |
| vLLM HiSparse 驻留哈希表探测 (`#61263`) | 平均 `3.40` 次探测/查询 | 平均 `1.15` 次探测/查询（负载 `<= 1/4`） | 哈希探测缩减 **`2.957x`** |
| TensorRT-LLM SID-GR 宽束搜索 Host 准备 (`#20830`, `W=64`) | `840.0 us/step` | `215.0 us/step` | **`3.907x`** 每步 Host 开销降低 |

## 3. 关联证据与后续可证伪问题

- **证据 ID**：`claim-sglang-qsa-indexer-rust-chat-dp-buckets-zimage-failclosed-vllm-megamoe-sp-mxfp4-hisparse-trtllm-phase99`
- **一手信源**：
  - `sglang-perf-qsa-optimize-prefill-indexer-in-long-cac-45791-642e9c1-2026`
  - `sglang-fix-rust-server-match-python-s-chat-layer-on-45659-4fb422b-2026`
  - `sglang-fix-dp-attention-align-prefill-graph-buckets-46138-9b230c1-2026`
  - `sglang-fix-diffusion-zimage-stop-serving-a-previous-46044-17204f3-2026`
  - `sglang-fix-diffusion-fail-closed-after-local-worker-46025-0f41fff-2026`
  - `vllm-perf-moe-size-the-deepgemm-megamoe-symmetric-63541-2feedbe-2026`
  - `vllm-kimi-k3-route-the-fp8-chunked-context-k-v-pac-61527-4eb5df8-2026`
  - `vllm-perf-mxfp4-optimize-activation-quant-for-mxfp-63101-f3bdb0e-2026`
  - `vllm-perf-hisparse-reduce-residency-hash-probes-61263-a2ecc8a-2026`
  - `tensorrt-llm-trtllm-16810-perf-sid-gr-3-n-reduce-per-step-20830-832b2cb-2026`
- **确切限制**：全部数据基于上游源码补丁分析与本地 Python 3.12 标准库确定性建模（`own-offline`）。
