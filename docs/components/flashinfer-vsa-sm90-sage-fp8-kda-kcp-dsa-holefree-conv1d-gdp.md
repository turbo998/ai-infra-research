# FlashInfer Hopper Cake VSA SM90 Sage-FP8 40 阶段稀疏注意力、KDA KCP `wave_fill1_p1tie` 浪涌规划与 Lean Arena、64 头 DSA 设备端 `-1` 空洞压缩与 Causal Conv1d+SiLU 6 路由

> **Phase 56 (`priority: 65`) · 一手源码与 Commit 级深挖**
> - `flashinfer-ai/flashinfer@8ec9464ce841bb75521303f827f41838bb28044e` (`#7092`)
> - `flashinfer-ai/flashinfer@baddd518c5433b39f21f9937e5888eb6212ac710` (`#7097`)
> - `flashinfer-ai/flashinfer@fb23d9702ce9e6a32506ec5329346894127a7a3e` (`#7084`)
> - `flashinfer-ai/flashinfer@462a48a4cd0fd91a98bc7dfdc218824c6da9f05e` (`#7098`)
> - `flashinfer-ai/flashinfer@69124ac26bd658fdba5e581ef51d0b5a55521303` (`#7089`)
> - `flashinfer-ai/flashinfer@8e8342f909352ec8cfdd351eb4759b5c8b5e351e` (`#6466`)
> - `flashinfer-ai/flashinfer@562917330df30b7a6ff79bbbd6a2d0baae749d80` (`#7086`)

---

## 1. Hopper (`sm_90a`) Cake VSA SM90 Sage-FP8 40 阶段可变块稀疏注意力 (`#7092`)

在 Hopper H100 (`sm_90a`) 架构上，`cake_vsa_sm90` 新增 19 个 FP8 镜像阶段（`vsa_fwd_k64_d2_fp8_sm90`, `vsa_fwd_k64_d3_fp8_sm90`, `vsa_fwd_k128_fp8_sm90`, `vsa_bwd_dq_fp8_sm90`, `vsa_bwd_dkv_*_fp8_sm90` 等）及 2 个量化算子（`vsa_q8_sm90`, `vsa_kv8_sm90`），将 `cake_jit.py` 的 `MODULES` 扩展至 **40 个阶段**。

- **Sage-FP8 E4M3 量化契约 (`_FP8_MAX = 448.0`, `_K_GROUP = 16`)**：
  - `q_scale` 形状为 `(1, H, Sq)`（按 token 粒度计算 `amax(dim=-1) / 448.0`）；
  - `k_scale` 形状为 `(1, H, ceil(Sk / 16))`（按 16-token 块粒度计算 `amax(dim=(-1, -2)) / 448.0`，并在加载时预乘 `sm_scale`）；
  - `v_scale` 形状为 `(H, 128)`（按通道维度 `amax(dim=-2) / 448.0`，延后至输出写回时做每列缩放）；
  - 在线 Softmax 矩阵乘法 $Q K^T$ 与 $P V$ 均运行于 Hopper WGMMA E4M3，其中概率矩阵 $P \in [0, 1]$ 通过平移行最大值 `max(m_new)` 映射至 `[0, 448]` 后舍入为 E4M3。
- **前向调度与共享内存分配**：当 `kv_tile == 128` 时启用 `k128_fp8`（384 线程，`163,840 B` SMEM）；当 `kv_tile == 64` 且 `active <= 18` 且 `q_blocks >= 32` 时启用双缓冲 `k64_d2_fp8`（`114,688 B` SMEM，提升每 SM 并发 CTA 驻留）；其余回退至三缓冲 `k64_d3_fp8`（`147,456 B` SMEM）。
- **作者 H100 80GB HBM3 基准 (`author-benchmark`, 冷 L2, 137 组形状)**：跨 `b20` (`1.0868x`)、`ext33` (`1.1537x`)、`real39` (`1.2238x`)、`long45` (`1.1542x`) 四套测试集，整体几何平均加速达 **1.1632x**（流程 A，136/137 胜出）与 **1.1602x**（流程 B，137/137 全胜），在 `h32-m512-n8192-k16` 处达到峰值 **1.6698x**（`0.0680 ms` vs `0.1135 ms`）。

---

## 2. Cake Packed KDA Context-Parallel (`KCP`) `wave_fill1_p1tie` 浪涌规划与 `KdaArena` Lean 工作区 (`#7097`, `#7084`)

- **KCP 8 阶段直通启动与 `wave_fill1_p1tie` 分段规则 (`#7097`)**：
  - 将 `kcp_fwd_o`（`kcp_sm_bounded`）与 `kcp_dav`（`kcp_pairs`: `(num_chunks_pad / 2, num_heads, 1)`）直接纳入 `PROGRAMS`（从 6 阶段扩充至 **8 阶段**），使前向输出与反向对角项在同一设备捕获流中直通发射。
  - 默认分段规划器 `P_RULE_DEFAULT = "wave_fill1_p1tie"` (`cake_kcp_plan.py`)：
    1. 当序列分片数 `n_frag <= 3` 且单分片最大块数 `NC_max_frag >= 257`（`P_chain >= 2`）时，最小化等分波次代价：
       $$\text{Cost}(P) = \left\lceil \frac{n_{\text{frag}} \cdot P \cdot H}{\text{SM}} \right\rceil \cdot \left\lceil \frac{\text{NC}_{\max}}{P} \right\rceil \cdot t_{\text{walk}} + (P - 1) \cdot t_{\text{compose}}$$
    2. 当 `n_frag in {2, 3}` 且 `NC_max_frag < 257` 且 `n_frag * H <= SM` 时，使用最长处理时间优先（LPT）评估各分片真实块数在 $\lfloor \text{SM}/H \rfloor$ 个槽位上的完工时间（makespan）：若 `P = 1` 的 LPT 完工时间不劣于基础分段数 $P$，则直接选取 **`P = 1`**，彻底消除 `kcp_affine_merge` 段间状态合成及反向重走开销。
  - **显存与 GB300 实测 (`author-benchmark`)**：`kcp_saved_bytes` 移除 `256 * T_r * H` 字节冗余行归属缓存（在 `T_r=16,384, H=64` 时每 Rank 节省 **256 MiB**）。在 NVIDIA GB300 (`sm_103a`, CP2/4/8/16) 的全部 144 组配置上相对 FLA KCP 取得 **1.4640x–2.7942x** 加速。
- **`KdaArena(lean=True, tail_passes=4)` 分批尾阶段复用 (`#7084`)**：将反向传播末端的 3 个按块并行阶段（`fused_grad`, `glm_intra`, `gate_epilogue`）沿 `NC_pad` 切分为 `B = tail_passes = 4` 个批次并复用 `v_new` 缓冲区，使反向工作区下限从 **4,488 B/token-head 降至 2,820 B/token-head**（缩减 **37.17%**），且梯度保持逐位一致（bitwise-identical）。

---

## 3. Cake 64 头 DSA 设备端 `-1` 空洞压缩与 Causal Conv1d+SiLU 6 路由 (`#7098`, `#7089`, `#6466`, `#7086`)

- **Cake 64 头 DSA 训练 (`#7098`)**：当 `gather_kv_indices` 的 256 个槽位中存在中间 `-1` 空洞（如固定步长采样或文档边界截断）时，旧实现退化为逐槽位遍历全部 256 项并为无效槽位发起冗余 TMA 传输。`#7098` 在 `index_prep` 内核（`compact=1, use_pdl=True`）中将每行有效索引紧缩至前缀 `0 .. k_count-1`、尾部补 `-1`，并在 `bwd_delta` 中直接按 `[512, Tq * 512, 1]` 步长原地读取 head-major `dout`。在 GB300 (`sm_103a`) 上使 75.8% 空洞行场景（`nope_docs264x992_rank0_fixedtail`）单步耗时从 `14.109 ms` 降至 `6.280 ms`（**2.247x**），GB200 (`sm_100a`) 上从 `13.964 ms` 降至 `6.352 ms`（**2.198x**）。
- **Causal Conv1d + SiLU 训练 6 路由 (`#7089`) 与 SM120 FP8 Prefill (`#6466`)**：`cake_causal_conv1d_train` 拆分为 `fwd_dt / fwd_bias / bwd_dt / bwd_bias / bwd_fp32g / bwd_bias_fp32g` 6 条专用路由并补齐 `C ∈ {1280, 1536, 2560}` 通道分块，在 GB300 上较 FLA 提速 **2.959x–6.649x**；`fmha_v2_prefill_deepseek` 在 SM120 上补齐 `D=256/256` 与 `D=128/128` FP8 E4M3-to-BF16 特化，在 GLM-4.7(-Flash) 上较 BF16 提速 **1.53x–1.61x**。

---

## 4. 确定性复现实验验证

运行 `python experiments/flashinfer-vsa-sm90-sage-fp8-kda-kcp-dsa-holefree-conv1d-gdp/run.py --check` 全部不变量通过（详见 `summary.json`）。
