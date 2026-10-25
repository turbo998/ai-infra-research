# TensorRT-LLM Rubin SM104 DeepSeek-V4/DSpark 稀疏索引融合与捕获收敛、vLLM D2D FlashInfer 预热缓存分发与二级 KV 存储失败背压、SGLang MiniMax-M3.1 NVFP4/MSA-v1 解码/HiCache 行拷贝/PD 分离 KV 签名/DP-Attention 对齐，以及 AI Infra 创新厂商单元经济学对比

> **语言 / Language**: [中文](trtllm-rubin-vllm-seed-sglang-msa-nvfp4.md) | [English](trtllm-rubin-vllm-seed-sglang-msa-nvfp4.en.md)
> **阶段 / Phase**: Phase 73 (`priority: 82`, Track B)
> **关联问题 / Question ID**: `q-phase73-trtllm-rubin-vllm-seed-sglang-msa-nvfp4`
> **证据等级 / Evidence Tiers**: `implementation` + `author-benchmark` + `own-offline` (`mastery: not-assessed`)

---

## 1. 概述与核心架构洞察

在 2026-10-24 至 2026-10-25 的上游演进中，`NVIDIA/TensorRT-LLM`、`vllm-project/vllm` 与 `sgl-project/sglang` 围绕 Rubin (`sm_104`) / Blackwell (`sm_100`/`sm_103`) 上的稀疏注意力（DeepSeek-V4 DSA / MiniMax MSA-v1）、跨节点预热与 KV 卸载容错、以及 PD 分离异构 TP 校验完成了多项关键工程闭环：

1. **TensorRT-LLM DeepSeek-V4 / DSpark Rubin (`sm_104`) 优化**（`TensorRT-LLM@d7e0ecac92`）：
   - **消除冗余 `latent_cache` 切片**：在 `attentionOp.cpp` 中，当启用 KV Cache Reuse 或 Chunked Context 时，`latent_cache` 并不会被下游消费；直接传 `std::nullopt` 省去了每次 step 构造切片视图的开销。
   - **`invokeIndexerKCacheQuantAndScatter` 融合量化与写入（含 `sm != 104` 硬件门控）**：在 `indexerKCacheGather.cu` 中，当 `is_IndexK_fp8 && head_dim == 128 && quant_block_size == 128` 且 `max_Num_Tokens_Per_Block <= 8`（每 Block 256 线程 = 8 Warps，每 Warp 负责 1 个 Token 的 FP8 E4M3 量化、Scale 写出与 `dst_k_ptr` Scatter）且 **`sm != 104`（排除 Rubin SM104）** 时，将两步 Kernel 合并为单次启动；当 `sm == 104` 或大批次每 Block Token 数 $> 8$ 时保留独立 `invokeFp8Quantize128` + `invokeIndexerKCacheScatter` 路径以避免寄存器/占用率退化。
   - **DSpark 投机捕获一次性收敛（`consolidate` 延后）**：在 `dspark.py` 中，将 `target_capture_export.consolidate()` 从每个目标捕获层内部移至全部 61 层遍历结束后的 `forward()` 尾部（从 $K_{\text{capture}} = 4$ 次同步拷贝降为 $1$ 次）。
2. **vLLM Daemon-to-Daemon (`SeedTransport`) FlashInfer Autotune 预热缓存分发与二级 KV 卸载失败背压**（`vllm@41dc477d24`, `vllm@2b1f75c80f`）：
   - **D2D 预热缓存分发与全组一致校验**：`SeedTransport` 区分未认证对端（整消息超时 `_REMOTE_REQUEST_TIMEOUT_S = 10.0s`、最大请求 `_MAX_REMOTE_REQUEST_SIZE = 64 KiB`，防止慢速滴流攻击阻塞串行 `accept` 循环）与已认证对端（`_AUTHENTICATED_TIMEOUT_S = 300.0s`）。`FlashInferWarmupCache` 按 `(kind, config_hash, dp_rank, world_rank)` 拉取 `autotune_configs_dp{dp}_rank{r}.json`，并在 `_load_shared_autotune_cache` 中通过 `all_gather_object` 要求 `tune_group` 全体 Rank 100% 命中同一指纹才跳过 Dummy-run 扫描（因为扫描内含集合通信，任一 Rank 未命中若单方面跳过将导致 NCCL 死锁）。
   - **二级存储层（`fs`/`obj`）写入失败背压**：修复了 `KVOffloadManager._process_finished_jobs` 此前仅在 `completed_job.success == True` 时更新背压检测器的漏洞（导致磁盘满 `ENOSPC`/`EDQUOT` 时每次写入都失败却永不触发背压，引发无限重试风暴）。新增 `EMABackpressureDetector.on_store_failed()`：在 `_STORAGE_TIER_TYPES = {"fs", "obj"}` 上连续失败达到 `DEFAULT_FAILURE_THRESHOLD = 3` 次时，将 EMA 抬升至 `_FAILURE_EMA_FACTOR = 4.0 * high_water_s`（`fs` 升至 `40 ms`，`obj` 升至 `200 ms`），使 `ThrottledDropPolicy(ramp_factor=2.0)` 100% 丢弃后续写入，直至空闲衰减放行单次探测写入。
3. **SGLang MiniMax-M3.1 NVFP4 + MSA-v1 解码 + HiCache `page_row_copy` + PD 分离 KV 签名 + DP-Attention 对齐**（`sglang@a99c7d050f`, `174bd2976d`, `5d11819f3a`, `ad950cdcd9`, `9b230c1dae`）：
   - **MiniMax-M3.1 NVFP4 混合精度栈與 `rcp.approx.ftz.f32` 位级对齐**：新增 `minimax_m3_nvfp4` 量化方法（MXFP8 投影 + NVFP4 路由专家 + FP8 注意力 + NVFP4 KV/Index-K，`group_size=16`，支持 `DSparkMiniMaxDraftModel` 稠密 FFN 草稿模型）。在 `nvfp4_store_kv_index.cuh` 中，每 Token 每头仅占 `64 B` E2M1 载荷 + `8 B` E4M3 缩放 = **`72 B`**（相对 BF16 `256 B` 实现 **`3.556x` 压缩**），并显式使用 `rcp.approx.ftz.f32` 而非 IEEE 除法以消除约 `0.1%` 的 E2M1 舍入边界偏差；在 `msa_v1/attn/dequant.cuh` 中当 CUDA $\ge 13.4$ 时使用 PTX `mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4` 直接将 4 个 E2M1 与 E4M3 Scale 相乘输出 FP8x4，并通过 `FlashInferQ8K8Decode` 零拷贝视图喂给 `trtllm_batch_decode_with_kv_cache`。
   - **HiCache `page_row_copy` 2-CTA PCIe 饱和拷贝**：仅用 `kPageRowBlockQuota = 2` 个 CTA（每 CTA 512 线程 = 8 个 64-lane Worker，每 Lane 并发 `kVecsPerLane = 16` 个 `16 B` `load_nc`/`store_nc` 事务）即可在 B300 PCIe 上跑出 **`51 GB/s` H2D 与 `52 GB/s` D2H**（对比 `8 / 32` vecs/lane 的 `46 / 35 GB/s`），把其余 `146–150` 个 SM 完全留给计算。
   - **PD 分离 `kv_layout_signature` 与 DP-Attention Bucket/SUM_LEN 对齐**：在 `sglang@5d11819f3a` 中，为每个请求的 Adler-32 KV Checksum 附带 64 位非零奇数布局签名 `blake2b(repr((page_size, kv_item_lens, state_item_lens)), digest_size=8) >> 1 | 1`，使异构 TP/PP 宽度的 Prefill→Decode 优雅跳过校验并告警一次，同时在开启 `--disaggregation-decode-enable-radix-cache` 时自动禁用全 Prompt 校验和；在 `sglang@9b230c1dae` 中将 DP-Attention Prefill CUDA Graph 的 `global_num_tokens` 统一向上对齐到同一捕获 Bucket（如 `[0, 310, 180, 420] -> [512, 512, 512, 512]`），并在 Decode 满足 `CustomAllReduceV2` 容量且无空闲 Rank 时优先走 `SUM_LEN`。

---

## 2. AI Infra 创新厂商单元经济学与架构权衡（Rule 13 专题）

结合 MiniMax-M3.1 NVFP4（`72 B/token/head` KV 缓存）与 Blackwell/Rubin 稀疏注意力演进，对比主流推理云（**Fireworks AI**、**Together AI**）与专用芯片架构（**Cerebras WSE-3**、**Groq LPU**）：

| 维度 | Fireworks AI / Together AI (Blackwell B200/GB300 GPU 云) | Cerebras Wafer-Scale Engine (WSE-3) | Groq LPU (TSP 确定性张量流架构) |
|---|---|---|---|
| **核心存储介质与带宽** | HBM3e (`8 TB/s` B200 / `288 GB` GB300) + PCIe Gen5/NVLink 主机分级缓存 | 晶圆级片上 SRAM (`44 GB` on-wafer, `21 PB/s` 带宽) | 片上 SRAM (`230 MB/chip`, `80 TB/s/chip`) + 多机架同步 C2C 互连 |
| **稀疏注意力与 NVFP4 KV 收益** | **极高**：`72 B/token/head` NVFP4 KV（`3.56x` 压缩）+ `page_row_copy`（2 CTA 跑满 `51–52 GB/s` PCIe）使单机并发上下文容量翻 3.5 倍 | **受限于片上总容量**：无需 HBM Offload，但超长上下文或多租户百并发 KV 池易触及片上 SRAM 容量天花板 | **极低单请求延迟、高固定芯片数成本**：适合短/中上下文极速生成（数百 tok/s），但在 128K+ 长上下文下 KV 状态需跨数十颗 LPU 分片 |
| **PD 分离与异构弹性** | 支持异构 Prefill TP 与 Decode DP/EP 独立扩缩容（通过 `kv_layout_signature` 与 Mooncake/NIXL 传输） | 晶圆级单一执行域，按 Whole-Wafer 粒度调度 | 编译器静态排定确定性流水线，动态变长稀疏 Top-K 路由收益受限于静态最坏情况预留 |

---

## 3. 可复现实验验证 (`own-offline`)

运行本仓库原创标准库仿真脚本验证全部 7 项不变量：

```powershell
python experiments\trtllm_rubin_vllm_seed_sglang_msa_nvfp4_sim\run.py --check
```

| 验证模块 | 关键不变量 / 指标 | 仿真验证结果 | 证据等级 |
|---|---|---|---|
| TensorRT-LLM Rubin `sm_104` Indexer Gather | `sm != 104 && max_tokens_per_block <= 8` 才启用融合内核；DSpark `consolidate()` 移出层循环 | `sm=104` 自动回退双内核；61 层 4 捕获点 `consolidate` 调用从 `4 -> 1` | `implementation` + `own-offline` |
| vLLM D2D `SeedTransport` & KV Offload 背压 | 未认证请求 `10s / 64 KiB` 上限；`tune_group` 必须 100% 一致命中；`fs`/`obj` 连续 3 次写入失败触发 `4.0x` 高水位 EMA | `7/8` 命中拒绝跳过；连续 3 次失败后 EMA 升至 `40 ms / 200 ms`，`ThrottledDropPolicy` 丢弃率 `100%` | `implementation` + `own-offline` |
| SGLang MiniMax-M3.1 NVFP4 + MSA-v1 + Disagg + DP | NVFP4 KV 每头每 Token `72 B`（`3.5556x` 压缩）；`kVecsPerLane=16` 达 `51/52 GB/s`；DP Prefill `[0,310,180,420] -> [512]*4` | 全部字节配比、`blake2b` 奇数签名与 DP Bucket 对齐 100% 吻合 | `implementation` + `author-benchmark` + `own-offline` |

---

## 4. 架构师练习与可证伪问题

1. **计算练习（vLLM 二级存储失败背压恢复探测）**：当本地文件系统层（`LOCAL_HIGH_WATER_S = 10 ms`）因 3 次连续 `ENOSPC` 将 EMA 置为 `40 ms` 后，在 `ThrottledDropPolicy(ramp_factor=2.0)` 下丢弃概率为 $P_{\text{drop}} = \min\left(1, \frac{\text{EMA} - H}{(R - 1)H}\right) = 1.0$。若后台清理释放了磁盘空间，系统如何跳出 100% 丢弃死锁？
   - *解答提示*：`EMABackpressureDetector` 在 `is_under_pressure()` 查询时会根据距离上次更新的空闲时间 $\Delta t_{\text{idle}}$ 按指数衰减 $\text{EMA}_{\text{eff}} = \text{EMA} \cdot 2^{-\Delta t_{\text{idle}} / \tau}$；当 $\text{EMA}_{\text{eff}} < 2.0 \times 10\text{ ms} = 20\text{ ms}$ 时，$P_{\text{drop}} < 1.0$，从而放行单次探测写入以刷新真实 I/O 延迟。
2. **可证伪假设（SGLang PD 分离 Radix Cache 与全量 KV Checksum 冲突）**：若在开启 `--disaggregation-decode-enable-radix-cache` 的同时强行启用全 Prompt Adler-32 校验，所有命中 Decode 端前缀缓存的请求都会报错中断，因为 Prefill 仅计算并发送后缀页面的 KV，而全量摘要无法与半截传输匹配。
