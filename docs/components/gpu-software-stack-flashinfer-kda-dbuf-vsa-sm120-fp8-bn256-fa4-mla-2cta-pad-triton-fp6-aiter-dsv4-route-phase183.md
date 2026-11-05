# GPU 软件栈：FlashInfer CuTe KDA 双缓冲屏障、Cake SM120 Block-64 VSA 稀疏注意力、SM107a 分组 FP8 `BLOCK_N=256` 与 216-SM `K1` 权重梯度、FlashAttention-4 2CTA MQA 头部内填充、Triton Gluon FP6 共享 Tile 与 AITER DeepSeek-V4.1 融合路由 `-1` 填充丢弃（Phase 183）

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-kda-dbuf-vsa-sm120-fp8-bn256-fa4-mla-2cta-pad-triton-fp6-aiter-dsv4-route-phase183.md)** | **[English](gpu-software-stack-flashinfer-kda-dbuf-vsa-sm120-fp8-bn256-fa4-mla-2cta-pad-triton-fp6-aiter-dsv4-route-phase183.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 183（`priority: 192`）审计了 Blackwell（`sm_100a` / B200 / GB200）、Blackwell Ultra（`sm_103a` / GB300）、RTX PRO 6000 Blackwell（`sm_120`）、Rubin（`sm_107a` / R200）与 AMD `gfx1250`（MI455X）算子与编译器栈的 **11 个上游主干提交**：

1. **FlashInfer CuTe DSL KDA 双缓冲 `update_ready` 屏障与 Cake SM120 `vsa_sm120_blk64` 持久化稀疏注意力**：
   - `0102aa9f9f46` (#8684) 修复 CuTe DSL KDA 分解链内核 `fused_kda_64_chain_k_dbuf_cute` 中单槽 `update_ready` 屏障的相位奇偶死锁（Warp 12 与 Warp 13/CG 在第 $c$ 块与第 $c+1$ 块上因相同 `phase` 竞争导致 `arrive_and_wait` 永久挂起）：将 `update_ready` 扩为 `2` 个屏障槽（`slot = chunk % 2`，`parity = (chunk // 2) % 2`），在 GB300 上即使向 Warp 12 注入 `2 ms` 延迟，全部 **`309 / 309`** 项参数化测试仍 100% 通过。
   - `7729459ffe1a` (#8799) 为 RTX PRO 6000 Blackwell（`sm_120`，188 SM）启用持久化 Cake `vsa_sm120_blk64` 块稀疏注意力后端（每 SM 驻留两个 128 线程 4-warp CTA 或单个 8-warp CTA，替代按输出块一维发射的 `vsa_attention_128x64`），在 22 个基准形状上取得 **`1.084x`** 几何平均加速，最高在 `tiny_s1024_k2` 上达 **`1.263x`**（`14.3 -> 11.4 us`），在长前缀 `minimaxh3_prefix_s109632_h7_p8_k64` 上达 **`1.229x`**（`6,879.9 -> 5,599.8 us`），全部 22 项正确性校验 `max_abs <= 1.95e-3`。
2. **FlashInfer SM107a 单程分组 FP8 `BLOCK_N=256`、216-SM `K1` 权重梯度、纯 FP32 SIMT 路由反向与 SM103a `H2048` 确定性对上限**：
   - `f0ec379398e0` (#8809) 将 R200（`sm_107a`，212/216 SM）上的单程 UE8M0 分组 FP8 GEMM 在 K 循环块数 `< 32` 时切换至 `BLOCK_N=256`（此前仅 `< 16` 启用），在 212-SM R200 的 `fib_q35b_gate_up_t4096`（`E=16, T=4096, N=1024, K=4096`）上将耗时从 `84.85 us` 降至 **`76.17 us`（`1.114x`）**。
   - `f8bedda6038a` (#8782) 在 216-SM `sm_107a` 芯片上恢复 `cake_dense_projection_gemm` 的 `K1` 权重梯度路径（`sm_count >= 208`），在 6 个代表性形状上实现权重梯度 **`1.06x–2.23x`** 加速（`4096->512, T=16173` 从 `81.58 -> 36.65 us` = **`2.23x`**）与完整反向 **`1.03x–1.71x`** 加速（`110.22 -> 64.59 us`）。
   - `8baa822f846e` (#8763) 在 148-SM `sm_100a`（GB200/B200）与 `sm_103a`（GB300）的 MoE 路由投影（`N <= 512`）反向中使用纯 FP32 FFMA SIMT 窗口（`simt_n128m64s2` / `simt_m32n64s8`）替代补偿式 Dot2/Kahan 累加，较补偿内核提速 **`1.24x–3.05x`**，较确定性 Cake 提速 **`1.04x–5.37x`**，且最坏相对 Frobenius 误差 $\le 2.1 \times 10^{-7}$（远低于 $2 \times 10^{-6}$ 预算）。
   - `386275f02fc6` (#8810) 将 GB300（`sm_103a`）`cake_rmsnorm_train` 的 `H=2048` 确定性配对反向窗口上限截断在 `T=58,496`（规避 `T=58,560` 时 `1.0622x` 的 `dw` 预算越界，将最坏比率压至 **`0.9454x`**），同时比非 Cake SOTA 快 **`1.1008x–1.3441x`**。
3. **FlashAttention-4 稠密 2CTA MQA 头部内填充、Triton Gluon FP6 共享 Tile、TileLang 零拷贝宿主映射与 ROCm AITER DeepSeek-V4.1 融合路由 `-1` 填充丢弃**：
   - `2ed4a1f0fc0e` (#3140) 在 FlashAttention-4 的稠密吸收式 MLA 中将非 2 的幂 MQA 头数在内核内填充至 `64` 或 `128`（如 `96 -> 128`、`48 -> 64`）并将 $\le 32$ 头路由至 1CTA，在 B200 上使 24 头预填充（`B=4, Sq=Sk=4096`）从 `4.38 ms` 降至 **`2.14 ms`（`2.047x`）**，96 头解码（`B=128, Sk=8192`）从 1CTA 的 `270.1 us` 降至 **`175.5 us`（`1.539x`）**。
   - `d86c8b2f0064` (#12548) 在 Triton Gluon 中为 Blackwell `tcgen05` 与 TMA 新增 FP6（`e2m3` / `e3m2`）共享内存 Tile 支持：TMA 载入采用 `b6x16_p32`（每 16 字节原子含 12B 紧缩数据 + 4B 填充）以喂入 `tcgen05_mma` B 操作数，TMA 存储采用 `b6p2x16` 将 128 通道紧缩为 96 字节全局内存，较 FP8 降低 **`25.0%`** 显存流量；`51f4f818afc1` (#12524) 为 AMD `gfx1250` `scaled_dot` 启用 LDS 流水预取；TileLang `fd5305807bfd` (#3802) 新增 `cuMemHostGetDevicePointer_v2` 宿主映射零拷贝缓冲区注解。
   - ROCm AITER `d0fd39399b46` (#7365) 修复 `gfx1250`（MI455X）FlyDSL 融合 TP 路由/部分和内核 `moe_route_psum_fused` 在 `topk_ids == -1`（动态专家激活掩码填充）时的无符号溢出越界故障，显式写入 `DROPPED_ROUTE_ROW` 哨兵并在 Stage 1 跳过无效行，在 4x MI455X 上运行 DeepSeek-V4.1-Flash + DSpark 推测解码完成 **`1,105 / 1,105`** 请求（**`0` 次内存故障**，修复前直接崩溃），实测吞吐达 **`420.0 tok/s/GPU`**（总吞吐 `1,680.1 tok/s`），TTFT p50 **`49.6 ms`**，TPOT p50 **`2.30 ms`**。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+--------------------------------------------------------------------------------------------------------+
| Phase 183 GPU Software Stack: Blackwell GB300/B200, RTX PRO 6000 (sm_120), Rubin R200 & MI455X gfx1250 |
+--------------------------------------------------------------------------------------------------------+
  [FlashInfer CuTe KDA Double-Buffer, Cake SM120 VSA & SM107a/SM103a GEMM/RMSNorm (#8684, #8799, #8809..)]
    |-- CuTe KDA Decomp (#8684)      --> 2-Slot update_ready (chunk%2, parity=(chunk//2)%2, 309/309 GB300)
    |-- Cake SM120 VSA (#8799)       --> Persistent vsa_sm120_blk64 (1.084x Geomean, up to 1.263x / 1.229x)
    |-- Cake SM107a FP8 GEMM (#8809) --> R200 Single-Run UE8M0 BLOCK_N=256 for K_blocks<32 (1.114x @ T=4096)
    |-- Cake 216-SM K1 Wgrad (#8782) --> sm_107a K1 Weight-Gradient Route (up to 2.23x wgrad / 1.71x bwd)
    +-- SIMT Router & Det-Pair (#8763)--> Plain FFMA Router Bwd (3.05x vs Comp) + GB300 H2048 Cap T<=58,496
                                      |
                                      v
  [FlashAttention-4 2CTA MQA Pad + Triton Gluon FP6 + TileLang + ROCm AITER DSV4 (#3140, #12548, #7365)]
    |-- FA4 2CTA MQA Pad (#3140)     --> In-Kernel Pad 96->128 / 48->64 & <=32h 1CTA (2.047x Prefill, 1.539x Dec)
    |-- Triton Gluon FP6 (#12548)    --> b6x16_p32 TMA Load + b6p2x16 TMA Store (96B/128 elems, -25% BW vs FP8)
    |-- TileLang Host-Mapped (#3802) --> cuMemHostGetDevicePointer_v2 Zero-Copy Host-Mapped Buffer Codegen
    +-- AITER DSV4 FlyDSL (#7365)    --> Drop -1 Pad Routes in moe_route_psum_fused (420 tok/s/GPU, 0 Faults)
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/flashinfer_kda_dbuf_vsa_sm120_fp8_bn256_fa4_mla_2cta_pad_triton_fp6_aiter_dsv4_route_phase183/run.py`](../../experiments/flashinfer_kda_dbuf_vsa_sm120_fp8_bn256_fa4_mla_2cta_pad_triton_fp6_aiter_dsv4_route_phase183/run.py) `--check` 100% 验证通过：
- `kda_update_ready_barrier_slots = 2`、`kda_gb300_passed_tests = 309`
- `vsa_sm120_geomean_speedup_22shapes = 1.084`、`vsa_sm120_max_speedup = 1.263`、`vsa_sm120_minimaxh3_speedup = 1.229`
- `cake_grouped_fp8_sm107a_bn256_speedup = 1.114`（`K_blocks < 32`）、`cake_sm107a_216sm_k1_wgrad_max_speedup = 2.23`
- `cake_simt_router_vs_comp_max_speedup = 3.05`、`cake_simt_router_vs_det_max_speedup = 5.37`、`cake_rmsnorm_sm103a_h2048_det_pair_cap_t = 58496`（`worst_dw_ratio = 0.9454`）
- `fa_mla_24h_prefill_speedup = 2.047`、`fa_mla_96h_decode_2cta_vs_1cta_speedup = 1.539`
- `triton_gluon_fp6_global_bytes_per_128_elems = 96`、`triton_gluon_fp6_vs_fp8_bw_saving_pct = 25.0`
- `aiter_dsv4_mi455x_tok_per_s_per_gpu = 420.0`、`aiter_dsv4_mi455x_ttft_p50_ms = 49.6`、`aiter_dsv4_mi455x_tpot_p50_ms = 2.30`（`1,105` 请求 `0` 故障）

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **GB300 / R200 混合注意力（KDA + MLA）与 MoE 路由训练** | 在 GB300 KDA 分解链启用双缓冲 `update_ready`（`#8684`），在 148-SM B200/GB300 MoE 路由反向启用纯 FP32 SIMT 窗口（`#8763`，最高 `3.05x`），在 212/216-SM R200 启用 `<32` K-block `BLOCK_N=256` 分组 FP8 GEMM（`#8809`）与 `K1` 权重梯度（`#8782`，`2.23x`）。 | 在 GB300 `H=2048` RMSNorm 反向 $T > 58,496$ 时继续使用确定性配对规约（`T=58,560` 会触发 `1.0622x` `dw` 预算越界，应切换至原子累加路径 `#8810`）。 |
| **Blackwell 吸收式 MLA 2CTA 解码与 RTX PRO 6000 VSA 稀疏注意力** | 对 `48 / 96` 头稠密吸收式 MLA 启用 FA4 2CTA 内核内填充（`#3140`，96 头解码 `1.539x`）并将 $\le 32$ 头路由至 1CTA（24 头预填充 `2.047x`）；在 `sm_120` 工作站 GPU 上启用持久化 `vsa_sm120_blk64`（`#8799`，长前缀 `1.229x`）。 | 对 $\le 32$ 头的稠密 MQA 强行使用 2CTA 填充（浪费一半集群 MMA 行宽），或在非 `sm_100a`/`sm_103a` 架构上调用 `tcgen05` `b6x16_p32` FP6 TMA 格式（`#12548`）。 |
| **AMD MI455X (`gfx1250`) DeepSeek-V4.1 动态专家路由推理** | 在 `gfx1250` 4x MI455X 生产部署 DeepSeek-V4.1-Flash + DSpark 时启用包含 `DROPPED_ROUTE_ROW` 负路由过滤的 FlyDSL `moe_route_psum_fused`（`#7365`，`420.0 tok/s/GPU`，TTFT `49.6 ms`，TPOT `2.30 ms`）。 | 将未经验证的 `topk_ids == -1` 填充路由直接强转为无符号索引写入 LDS `expert_Token_num`（会导致 Stage 1 越界写及随机 GPU 内存页故障）。 |
