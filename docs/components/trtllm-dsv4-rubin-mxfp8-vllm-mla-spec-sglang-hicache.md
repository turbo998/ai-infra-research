# TensorRT-LLM DeepSeek-V4 Rubin SM107 与 CuteDSL MoE、vLLM MLA/自适应投机采样、SGLang HiCache 及 AI Infra 创新公司（Fireworks/Together/Cerebras/Groq）深度解析

> **English Version**: [trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache.en.md](trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache.en.md)  
> **Phase**: 72 (`priority: 81` · Track B) · **核验日期**: `2026-10-25` (UTC+08:00)  
> **实验脚本与密封输出**: [`experiments/trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache/run.py`](../../experiments/trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache/run.py) · [`summary.json`](../../experiments/trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache/summary.json)

---

## 1. 背景与核心架构问题

在超大规模稀疏 MoE 模型（如 DeepSeek-V4 / DSpark、Muse Spark 816B-A42B、GLM 5.3、Gemma4 多模态）的生产推理栈中，TensorRT-LLM、vLLM 与 SGLang 近期围绕 **NVIDIA Vera Rubin (`sm_107`) / Blackwell (`sm_100`/`sm_103`)**、**投机解码 (Speculative Decoding)** 与 **分层 KV Cache (HiCache)** 落地了关键优化：

1. **Rubin SM107 上 DeepSeek-V4 MLA 投影与旋转位置编码的冗余搬运**：在传统 MLA 路径中，`q_b_proj` 输出 BF16，随后单独分配 `q_pe` 并启动 RoPE + FP8 量化内核，在 `o_b_proj` 阶段还需额外分配中间输出张量再拷贝回主干。
2. **MoE 输入异步量化的 Host 同步开销与 `NVFP4_AWQ` 预缩放缺失**：在 GB300 小 batch (`batch=8`) 解码中，将 MoE 输入量化放到辅助流（aux stream）与 top-k 路由重叠可节省约 `25 us/layer`，但在 Eager 单流模式下，流切换与事件同步的 CPU 开销反而超过收益；同时，流式加载 `NVFP4_AWQ`/`NVFP4_ARC` 专家权重时若遗漏 `pre_quant_scale` (`fc31_act_scale`)，会导致激活在未平滑状态下直接按平滑后的 `input_scale` 量化，引发严重数值失真。
3. **vLLM Batch-Sharded Sampling 与 GPU 自适应投机验证的张量形状失配**：vLLM 的 `BatchSharder` 在 CPU 上根据调度的 draft token 上界划分 all-to-all 采样计划，而 `AdaptiveVerificationManager` 在 GPU 上动态裁剪每个请求实际准入的 draft token 数，导致开启自适应验证时无法启用分布式采样分片。
4. **SGLang PD 分离解码下 EAGLE 二元组 (Bigram) 树缓存恢复的边界截断**：EAGLE 投机树以 $(N)$ 个 bigram 键索引 $(N+1)$ 个 token ID。当 PD Decode 节点从 HiCache 恢复前缀时，若仅按 `decode_prefix_len` 匹配，会漏掉末尾第 $(N+1)$ 个边界 token。
5. **AI Infra 创新公司架构权衡（Fireworks / Together vs Cerebras / Groq）**：面对 400B–800B+ 总参数、30B–45B 激活参数的稀疏 MoE 与投机解码工作负载，基于片上/晶圆级 SRAM 的确定性数据流架构（Groq LPU、Cerebras WSE-3）与基于 HBM3e/HBM4 + NVFP4/MXFP8 + 分布式投机验证的云端软件栈（Fireworks AI、Together AI）呈现出截然不同的容量、延迟与 TCO 边界。

---

## 2. 一手源码审计与实现机制

### 2.1 TensorRT-LLM `d7e0ecac925d` (`#19961`), `032eb456ecb1` (`#20808`), `ecae3c97e2b6` (`#20823`), `abb94ab8fc1c` (`#20663`) & `9666d7427a8c` (`#20774`)

1. **DeepSeek-V4 / DSpark 在 Rubin (`sm_107`) 上的端到端融合 (`#19961`)**：
   - **默认启用 CuteDSL Block-Scaling GEMM**：在 `modeling_deepseekv4.py` 的 `get_model_defaults` 中设置 `"use_cute_dsl_blockscaling_mm": get_sm_version() == 107`。
   - **`q_b` 融合旋转 FP8 Q (`q_rope_applied = True`)**：在 `cpp/tensorrt_llm/thop/attentionOp.cpp` 中，新增判定 `mla_params.q_rope_applied = mla_params.fuse_q_fp8_in_rope && !q_pe.has_value()`。当上游 DSv4 `q_b` GEMM 融合内核已直接将旋转后的完整 FP8 Q 写入 `quant_q_buffer` 时，不再传入 `q_pe`，跳过 `mla_rope_generation` 中对 RoPE 段的二次改写；同时在模块初始化时预注册持久化 buffer `_quant_scale_qkv = torch.ones(1, dtype=torch.float32)`，消除首次 CUDA Graph 捕获时的 H2D 拷贝。
   - **`o_b_proj` 原地 MXFP8 GEMM (`allow_unpartitioned_output = True`)**：在 `CuteDSLBlockScaledRubinLinear` 中支持无 `partition_id` 的原地输出写入，`_rubin_dsv4_o_b_proj_out` 直接把 SM107 MXFP8 GEMM 结果写入调用方 `output`。按 `128` 头、`q_dim=192`、`hidden=7168` 计算，每 token 消除 `16,384 B` (`q_pe`) + `49,152 B` (BF16 Q 占位读写) + `14,336 B` (`o_b_proj` 中间缓冲) = **`79,872 B/token`** 的冗余显存流量。
   - **CuteDSL NVFP4 MegaMoE 投机接受长度基线 (`acceptance_length.yaml`)**：新增 `TestDeepSeekV4ProDSpark::test_gsm8k_dep4_megamoe_cutedsl`，实测平均接受长度 `ref_al = 4.433448`（下限 `min_al = 4.211776`，GB300 GSM8K 准确率 `96.21%`），与 FP8 DeepGEMM DEP8 基线 (`ref_al = 4.448785`) 保持同级精度。
2. **CuteDSL MoE 输入异步量化的多流门控 (`#20808`)**：
   - 在 `moe_scheduler.py` 中新增 `_should_quantize_input_async(x)`：仅当后端支持 `quantize_input_async`、无外部通信接管 (`comm is None`)、无输入侧路由权重缩放 (`not apply_router_weight_on_input`) **且** 处于 `do_multi_stream()`（CUDA Graph Runner）或 `torch.cuda.is_current_stream_capturing()` 时才分叉到辅助流。在 GB300 `batch=8` 下每层重叠 `~25 us`（61 层累计节省 **`1.525 ms/step`**），同时避免 Eager 模式下的 CPU 同步退化。
3. **MegaMoE / CuteDSL / CUTLASS `NVFP4_AWQ` 预缩放因子修复 (`#20823`)**：
   - AWQ/ARC 校准的 NVFP4 权重存储为 $W_{\text{stored}} = W / s$，且 `fc31_input_scale` 是基于平滑后的激活 $x \odot s$ 标定的。`#20823` 在 `_load_nvfp4_expert` 流式加载器以及 `mega_moe_cute_dsl.py`、`fused_moe_cute_dsl.py`、`w4a16_nvfp4.py` 等 6 个后端的 `quantize_input` 入口统一执行 `x = x * self.fc31_act_scale`，修复了漏乘导致的 `233%` 相对内积误差（见合成算例）。
4. **Gemma4 多模态 Encoder 预算调度 (`#20663`) 与 LoRA 格式重載 (`#20774`)**：
   - `#20663` 为 `Gemma4InputProcessor` 实现 `get_mm_encoder_item_metadata` 与 `_padded_patches_per_frame = max_soft_tokens * pooling_kernel_size^2`（`280 * 3^2 = 2520` patches/frame，音频 4x 下采样），使多模态 chunked prefill 严格受 `max_num_encoder_tokens` 预算约束；而无独立视觉塔的 `Gemma4UnifiedInputProcessor` 显式关闭 encoder 预算。
   - `#20774` 在 `LlmRequest` 中持久化 `py_lora_ckpt_source`，确保 `PeftCacheManager.add_request_peft` 在缓存未命中从磁盘重载时使用请求入队时验证过的 `ckpt_source`（如 `"hf"` vs `"nemo"`），而非回退到服务器全局默认格式。

---

### 2.2 vLLM `56e4e42868b4` (`#62102`), `7050ef881ff0` (`#58466`) & `5f63ea461ccd` (`#63001`)

1. **`concat_and_cache_mla` 单位缩放因子编译期特化 (`#62102`)**：
   - 在 `csrc/cache_kernels.cu` 中为 `concat_and_cache_mla_kernel` 增加模板布尔参数 `bool skip_scale`。当 `kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2")` 且 `self._k_scale_float == 1.0` 时，内核直接调用 `fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], 1.0f)`，省去设备端 `*scale` 全局内存解引用与除法/乘法路径，在 GLM 5.3 / DeepSeek-V3.2 上实测带来 **1.2% ~ 3.0% 端到端 TPOT 改善** (`author-benchmark`)。
2. **Batch-Sharded Sampling 兼容 GPU 自适应投机验证 (`#58466`)**：
   - 在 `AdaptiveVerificationManager` 中新增 `AdmittedDraftCounts`（由页锁定主机张量 `_admitted_drafts_cpu` 与 `torch.cuda.Event` 组成）。当 `reallocate_drafts` 在 GPU 上按置信度曲线重新分配各请求的准入 draft 数后，异步 D2H 拷贝至主机；`BatchSharder.shard_sampler_inputs` 等待该事件并用真实准入计数 `1 + num_admitted_draft_tokens_np` 替代上界 `cu_num_logits_np`，使 TP 各 rank 构造出零填充浪费的精确 all-to-all 采样分片计划。
3. **ModelOpt 混合精度 DFlash / DSpark 投机草稿层名映射 (`#63001`)**：
   - 新增 `map_draft_quant_config`：先应用 `DFlashQwen3Model.hf_to_vllm_mapper`（将 `midlayer` 映射为 `layers.0`），再按**逆序** `reversed(range(num_hidden_layers))` 将 `layers.{i}` 偏移为 `layers.{start_layer_id + i}`（防止 `start_layer_id=1` 时发生 `0 -> 1 -> 2` 级联覆盖），并补齐运行时前缀与 ROCm MI300 `E4M3FN -> E4M3FNUZ` 归一化。

---

### 2.3 SGLang `0ff67ee72aaf` (`#45076`), `f82af9ca17c9` (`#40595`) & `8ef5dec95faa` (`#46047`)

1. **PD Decode HiCache 恢复中的 EAGLE Bigram 边界 Token 修正 (`#45076`)**：
   - EAGLE 树的每个节点键存储二元组（$N$ 个 bigram 需要 $N+1$ 个 token ID）。在 `DecodeHiCachePreallocMixin._try_hicache_queue_load_back` 中，将 `match_kv_cache` 的 `max_prefix_len` 修正为 `pm.decode_prefix_len + int(tree_cache.tree_core.is_eagle)`，修复了恢复路径少匹配 1 个边界 token 的 bug。
2. **`UnifiedRadixCache` 动态外部直连存储插件 (`#40595`)**：
   - 新增 `--unified-cache-external-linker-backend=dynamic` 与 `--unified-cache-external-linker-config`（JSON 指定 `linker` 类名、`linker_module_path` 与 `linker_extra_config`），允许在不经过主机 DRAM 缓存层的情况下将 `UnifiedRadixCache` 直接挂接到外部定制 KV 存储（与内置 `mooncake`、`mori` 并列）。
3. **视频扩散模型 (WanVideo / Cosmos3 / MOVA) 每步 Host Sync 消除 (`#46047`)**：
   - 将 RoPE `cos_sin_cache` 与条件掩码按 `(items, device, dtype)` 缓存在 `forward_batch.extra` 中，并将 `cosmos_dreams` 的 `null_action_token_positions` 从 Python `list[int]`（每次索引触发阻塞式 H2D 拷贝）改为连续 `list[slice]`，使每个去噪步的 CPU-GPU 同步次数降为 `0`。

---

## 3. AI Infra 创新公司专题：Fireworks / Together vs Cerebras / Groq 在超大 MoE 与投机解码时代的架构经济学

结合本轮 TensorRT-LLM (`#19961` DSpark/DeepSeek-V4 on Rubin/GB300) 与 vLLM (`#58466`, `#63001` DFlash/DSpark 自适应投机解码) 的最新进展，我们对四家代表性 AI Infra 创新公司的技术路线进行量化对比：

| 维度 | Fireworks AI (FireAttention / Disagg SpecDecode) | Together AI (Turbo / Speculative MoE Cloud) | Cerebras (WSE-3 Wafer-Scale Engine) | Groq (LPU v1/v2 Deterministic TSP) |
|---|---|---|---|---|
| **核心硬件载体** | NVIDIA GB200/GB300 NVL72 & Vera Rubin (`sm_100a`/`103a`/`107a`) | NVIDIA H200 / B200 / GB300 集群 + FlashAttention-3/4 & TileLang/CuteDSL | 单晶圆 WSE-3 (`900,000` 核心, `44 GB` 片上 SRAM + MemoryX 机柜) | 自研 ASIC LPU (`230 MB` 片上 SRAM/芯片, 80 TB/s 片上带宽) |
| **816B-A42B / DeepSeek-V4 级 MoE 适配性** | **极高**：NVFP4 下 816B 总权重仅需 `~408 GB`（单台 4xGB300 即可全量驻留 HBM3e），每步仅读取 `42B -> 21 GB` 激活专家 | **极高**：结合 ModelOpt 混合精度 DFlash/DSpark 草稿 (`AL ≈ 4.43`) 与 Batch-Sharded Sampling，高并发吞吐成本最优 | **中等**：`44 GB` 晶圆 SRAM 无法装下 400B+ 总权重，依赖 MemoryX 权重流式注入或跨晶圆流水线，稀疏跳过收益受限于流式总线 | **受限**：单芯片 `230 MB` SRAM 仅适合 8B–70B 稠密模型；816B 权重需上千张 LPU 静态切分，机柜间通信与硬件成本急剧膨胀 |
| **单流超低延迟 (TPOT)** | GB300 NVFP4 理想带宽下 `21 GB / 8 TB/s ≈ 2.63 ms/step`，配合 DSpark (`AL=4.43`) 等效单流可达 **~1,200+ tok/s** | 与 Fireworks 同属 HBM + NVFP4 + 投机验证路线，长上下文通过 MLA FP8 (`skip_scale`) 与 HiCache 保持低 TTFT | 单流 `batch=1` 峰值可达 **1,500–2,000+ tok/s**（当模型或激活子集完全命中的晶圆 SRAM 时） | 70B 稠密模型单流可达 **300–800+ tok/s** 且抖动极低（编译器静态调度，无缓存未命中） |
| **架构师选型结论** | **超大稀疏 MoE + 多租户 Agentic 长上下文首选**（复用 Rubin/Blackwell NVFP4、PD 分离与 DSpark 投机树） | **开源模型快速定制、RL 后训练推理一体化与混合精度投机部署首选** | **固定超大批量代码生成/科学推理或晶圆可容纳模型的极致单流吐字场景** | **中小参数模型 (`<=70B`)、实时语音/金融低抖动确定性 SLA 场景** |

---

## 4. 合成验证实验与架构师练习

运行本组件配套的验证脚本：

```bash
python experiments/trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache/run.py --check
```

### 架构师设计练习与可证伪问题

1. **计算题**：在 DeepSeek-V4 (`61` 层 MoE + MLA) 的 GB300 解码中，若 CUDA Graph 多流模式下每层通过 `_should_quantize_input_async` 将 NVFP4 输入量化与 `moe_sort` 重叠节省 `25 us`，且 DSpark 投机解码平均每步接受 `4.433` 个 token，求每生成 `1,000` 个输出 token 累计节省的 GPU 墙钟时间。
   - *解答*：单步（验证 1 次）节省 $61 \times 25\ \mu\text{s} = 1.525\ \text{ms}$；生成 1,000 个 token 需要 $1000 / 4.4334 \approx 225.56$ 步，累计节省 $225.56 \times 1.525\ \text{ms} \approx \mathbf{344.0\ \text{ms}}$。
2. **可证伪假设**：对 `NVFP4_AWQ` 导出的 MoE 专家权重，若在 `quantize_input` 阶段省略 `x * fc31_act_scale` 而在 GEMM 输出端乘回平均缩放因子，能否恢复正确输出？
   - *证伪路径*：`fc31_act_scale` 是逐通道（per-channel, `[1, hidden]`）平滑向量，位于内积求和维度内部：$\sum_j (x_j s_j)(W_{ij}/s_j) \neq (\bar{s})\sum_j x_j (W_{ij}/s_j)$，且 FP4 动态范围截断发生在逐通道缩放之后，事后标量补偿在数学上不成立（实验脚本验证漏乘导致 `233.33%` 误差）。
