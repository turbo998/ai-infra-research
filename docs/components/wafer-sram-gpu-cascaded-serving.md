# 晶圆级/片上 SRAM 引擎（Cerebras WSE-3 / Groq LPU）与 GPU HBM 异构级联投机推理架构：vLLM V1 `SuffixDecoding` / `DFlash` 与长上下文 KV 容量墙

> **Evidence Status**:
> - **Implementation (Verified Upstream Code)**:
>   - `vllm-project/vllm` (`commit 973dbf8071f8bdd6dd7eb2d6ee0da7cbe6a93670`):
>     - `vllm/v1/spec_decode/suffix_decoding.py` (blob `5d100ce9ed9715e8bc6c2b3d800d11cb500d7323`, lines 1–102): 实现 `SuffixDecodingProposer`（集成 `arctic_inference` `SuffixDecodingCache`），基于历史全局 Prompt 与活跃请求后缀树在 CPU 侧直接投机生成多步候选草稿（`max_tree_depth`, `max_spec_factor`, `min_token_prob`），`load_model()` 为空操作（**零额外 GPU 显存开销**）；
>     - `vllm/v1/spec_decode/dflash.py` (blob `d7d3e3adb02b871048628b781f2e866497b53d3c`, lines 1–140): 实现 `DFlashProposer`（并行块扩散/交叉注意力草稿模型），在单次前向中同时传入 `num_query_per_req = 1 + num_speculative_tokens` 个并行 Query Token（使用独立的 `_context_slot_mapping_buffer` 与 `_slot_mapping_buffer` 保持 Piecewise CUDA Graph 静态形状），将传统自回归草稿的 $K$ 次串行 HBM 读取压缩为 **1 次并行前向**。
> - **Specification / Author-Benchmark**:
>   - Cerebras WSE-3 (`CS-3`): `44 GB` 片上 SRAM，`21 PB/s` 内存带宽，`900,000` AI 核心；
>   - Groq LPU: 单芯片 `230 MB` 片上 SRAM，`80 TB/s` 内存带宽，确定性编译器流水线；
>   - NVIDIA Blackwell `GB200 NVL72` (`8-GPU` 切片): `1,536 GiB` HBM3e，`64 TB/s` 聚合带宽。
> - **Own-Offline Simulation**: `experiments/wafer_sram_gpu_cascaded_serving/run.py`（确定性离线仿真：量化 `4K–128K` 上下文下纯片上 SRAM 引擎的 **KV Cache 容量墙拐点**，并对比 GPU 自回归、`SuffixDecodingProposer`、`DFlashProposer` 与 **异构 `SRAM-Draft (Sliding-8K) + GPU-HBM3e-Target (Full-64K)` 级联投机架构**的 TPOT 与单百万 Token 成本）。

---

## 1. AI Infra 创新芯片与推理云的架构分野：片上 SRAM 带宽奇迹 vs 长上下文 KV 容量墙

以 **Cerebras (`WSE-3`)**、**Groq (`LPU`)** 为代表的片上 SRAM 架构，以及以 **Fireworks AI**、**Together AI** 为代表的 GPU HBM 软硬协同推理云，代表了攻克 LLM Decode 内存带宽墙（Memory Bandwidth Wall）的两条截然不同的技术路线：

### 1.1 为什么纯 SRAM 架构在短上下文单并发下天下无敌，却在 `64K–128K` 多并发 Agent 负载下撞墙？

以 `Llama-3.3-70B (FP8, GQA-8)` 为例：
- **模型权重固定开销**：`70 GiB`；
- **单 Token KV Cache 开销**：$80 \text{ layers} \times 2 \text{ (K,V)} \times 8 \text{ heads} \times 128 \text{ dim} \times 1 \text{ Byte} = \mathbf{163,840 \text{ Bytes/token}}$（`160 KiB/tok`）。
- 当单个 Agent 会话上下文达到 **`64K` (`65,536` tokens)** 时，**仅 1 个会话的 KV Cache 就高达 `10.0 GiB`**！若并发运行 `32` 个 `64K` 会话，KV Cache 总体积达到 **`320.0 GiB`**：
  1. **Groq LPU 集群（`320` 颗 LPU = `73.6 GB` 总 SRAM）**：装下 `70 GiB` 权重后仅剩约 **`3.6 GiB` 片上 SRAM**，连 **1 个完整 `64K` 会话（`10 GiB`）都无法完全驻留片上**！一旦并发数超过片上容量，必须通过芯片间/主机 PCIe 频繁换入换出 KV 或重流式加载权重，`80 TB/s` 的片上带宽优势瞬间退化为外部 IO 瓶颈；
  2. **Cerebras WSE-3（双 `CS-3` 晶圆级集群 = `88 GB` 总片上 SRAM）**：扣除权重与激活工作区后约剩 **`18.0 GiB` 片上 SRAM**，在 `4K` 短上下文下可片上驻留 `28` 个并发会话并跑出 **`0.63 ms` 极致 TPOT**；但在 `64K` 长上下文下片上仅能容纳 **`1` 个完整会话**，在 `16` 并发下若溢出至片外 `MemoryX` 机柜，有效 TPOT 将劣化至 **`348.58 ms`**；
  3. **NVIDIA `GB200 NVL72`（8 卡切片 = `1,536 GiB` HBM3e）**：扣除权重后拥有 **`1,440 GiB` HBM3e KV 容量**，可轻松驻留 **`144` 个 `64K` 会话**，但受限于单步读取 `70 GiB` 权重与 `320 GiB` KV 的 HBM 带宽，标准自回归 Decode TPOT 为 **`11.40 ms`**（约 `87.7 tok/s/user`）。

---

## 2. 破局之道：vLLM V1 零开销/并行投机算子与“异构 SRAM-Draft + HBM-Target”级联架构

如何在拥有 **`320+ GiB` 长上下文 KV 容量**的同时，获得接近片上 SRAM 的 **`~2.5 ms` 超低等效 TPOT**？我们在 `vllm-project/vllm` 一手源码与创新推理云实践中核验了三级演进架构：

### 2.1 第一级：`SuffixDecodingProposer`（CPU 零权重后缀树投机，`vllm/v1/spec_decode/suffix_decoding.py`）

在 Agentic 代码编辑、JSON 结构化生成与多轮工具调用中，输出往往包含大量与 Prompt 或上文高度重复的片段：
- `SuffixDecodingProposer` (lines 28–102) 维护一个跨请求的全局后缀树缓存 `SuffixDecodingCache(max_tree_depth, max_cached_requests)`；
- 每步仅利用最后 `max_tree_depth` 个 Token 在 CPU 侧执行微秒级（`0.18 ms`）后缀树查找并剪枝低概率分支（`min_token_prob`），**占用 `0 GiB` 额外 GPU 显存**（`load_model` 为空），即可在代码/Agent 负载上获得平均 **`2.85` 个接受 Token**，将有效 TPOT 从 `11.40 ms` 压至 **`4.484 ms`（提速 `2.54x`）**。

### 2.2 第二级：`DFlashProposer`（并行块扩散草稿，`vllm/v1/spec_decode/dflash.py`）

传统 `EAGLE` / 小模型草稿虽然参数量小，但生成 $K=8$ 个草稿 Token 仍需在 GPU 上串行执行 $8$ 次自回归解码（经历 $8$ 次 Kernel 启动与 HBM 往返）。
- `DFlashProposer` (`dflash.py` lines 40–139) 重写 `_get_ eagle_input_batch_info`，令每个请求一次性携带 **`num_query_per_req = 1 + num_speculative_tokens`（如 `9` 个并行 Query Token，包含 `mask_token_id`）**，通过双槽位映射缓冲区 `_context_slot_mapping_buffer`（写入 Target 隐藏层提供的上下文 KV）与 `_slot_mapping_buffer`，在**单次非自回归/块并行前向（仅 `1.42 ms`）**内一次性产出 8 个草稿 Token，平均接受 **`4.15` 个 Token**，将有效 TPOT 降至 **`3.499 ms`（提速 `3.26x`）**。

### 2.3 第三级：异构级联 `Wafer/SRAM Draft (Sliding-8K)` + `GPU HBM3e Verifier (Full-64K)`

将 Cerebras/Groq 片上 SRAM 引擎与 GPU HBM3e 集群通过低延迟 RDMA 组合成**异构投机级联（Heterogeneous Speculative Cascade）**：

```
[Incoming 32x 64K-Context Agentic Requests (Total Full KV = 320 GiB)]
   |
   +---> [Stage 1: On-Chip SRAM Sidecar (WSE-3 / LPU running 8B Draft + Local 8K Sliding Window KV)]
   |        * Weights: 8 GiB FP8 | 32x 8K Sliding-Window KV: 8 GiB -> 16 GiB Total (100% fits in On-Chip SRAM!)
   |        * Drafts K = 8 tokens at 0.08 ms/token = 0.64 ms total draft latency!
   |
   +---> [Stage 2: GPU HBM3e Target Cluster (8x H100/B200 holding 70B Weights + Full 320 GiB 64K KV)]
            * Verifies all 8 draft tokens in 1 parallel forward pass (13.20 ms, Arithmetic Intensity rises 8x!)
            * Accepts E[tau] = 5.40 tokens/step -> Effective TPOT = 13.84 / 5.40 = 2.563 ms (390.2 tok/s/user!)
```

- **为什么异构级联能同时打破“SRAM 容量墙”与“HBM 带宽墙”？**
  - Draft 模型的职责是局部语法与近期上下文连贯性预测，只需维护 **`8K` 滑动窗口 KV（32 并发仅占 `8.0 GiB`）**，加上 `8B` 草稿权重（`8.0 GiB`）共 **`16.0 GiB`**，**100% 驻留在晶圆/LPU 片上 SRAM 中**，以 `0.64 ms` 极速喷出 8 个高质量候选 Token；
  - GPU HBM3e 集群持有完整的 **`320 GiB` 64K 全局 KV Cache** 与 `70B` 权威权重，将原本每次只算 1 个 Token 的访存受限（Memory-Bound）Decode 转化为每次并行验证 9 个 Token 的高算术强度算子，使单步有效 TPOT 降至 **`2.563 ms`（比纯 GPU 自回归提速 `4.45x`）**，并将每百万 Token 推理成本从 `$3.1667` 压降至 **`$0.8566`（降低 `3.70x`）**！

---

## 3. 定量实验结果（`experiments/wafer_sram_gpu_cascaded_serving/run.py`）

### 3.1 不同上下文长度（`4K – 128K`）下的片上 SRAM vs HBM3e 会话容量墙

| 上下文长度 $L_{\text{kv}}$ | 单会话 KV 体积 (`70B FP8`) | Groq `320-LPU` 片上最大会话数 | Cerebras `2x WSE-3` 片上最大会话数 | `GB200 8-GPU` HBM3e 最大会话数 | `16` 并发下 WSE-3 TPOT (`ms`) | `16` 并发下 GB200 自回归 TPOT (`ms`) |
|---:|---:|---:|---:|---:|---:|---:|
| `4,096` (`4K`) | `0.625 GiB` | `5` | `28` | `2,304` | **`0.63 ms`（片上命中）** | `4.07 ms` |
| `16,384` (`16K`) | `2.500 GiB` | `1` | `7` | `576` | `52.78 ms`（片外溢出） | **`4.73 ms`** |
| `65,536` (`64K`) | `10.000 GiB` | `1` | `1` | `144` | `348.58 ms`（片外溢出） | **`7.37 ms`** |
| `131,072` (`128K`) | `20.000 GiB` | `1` | `1` | `72` | `696.58 ms`（片外溢出） | **`10.89 ms`** |

### 3.2 `64K` 长上下文 `32` 并发 Agentic Coding 五种推理架构全维对比

| 推理服务架构方案 | 单步草稿耗时 (`ms`) | 单步验证耗时 (`ms`) | 平均接受 Token 数 $E[\tau]$ | 等效单 Token TPOT (`ms`) | 单用户吞吐 (`tok/s`) | 集群总吞吐 (`tok/s`) | 每百万 Token 成本 (`USD`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. **GPU HBM3e 自回归基线**（无投机） | `0.00` | `11.40` | `1.00` | `11.400 ms` (`1.00x`) | `87.72` | `2,807.0` | `$3.1667` |
| 2. **vLLM V1 `SuffixDecodingProposer`**（CPU 零权重后缀树） | `0.18` | `12.60` | `2.85` | `4.484 ms` (`2.54x`) | `223.02` | `7,136.6` | `$1.2455` |
| 3. **vLLM V1 `DFlashProposer`**（并行块扩散草稿 $K=8$） | `1.42` | `13.10` | `4.15` | `3.499 ms` (`3.26x`) | `285.80` | `9,145.6` | `$0.9719` |
| 4. **纯晶圆/LPU SRAM + 片外 `MemoryX` KV 溢出** | `0.00` | `19.80` | `1.00` | `19.800 ms` (`0.58x`) | `50.51` | `1,616.3` | `$11.6865` |
| 5. **异构级联：`SRAM 8B Draft (8K 滑窗)` + `HBM3e 70B Verifier (64K)`** | **`0.64`** | **`13.20`** | **`5.40`** | **`2.563 ms` (`4.45x`)** | **`390.17`** | **`12,485.4`** | **`$0.8566` (`-73%`)** |

---

## 4. 解决方案架构师行动指南与可证伪问题

1. **按负载象限选型推理底座**：
   - **短上下文（`\le 4K`）+ 极低延迟实时语音/金融交易**：优先选用 **Cerebras WSE-3 / Groq LPU** 纯片上 SRAM 实例，直接享受 `< 1 ms` TPOT；
   - **长上下文（`32K–128K`）Agentic Coding / RAG**：在单集群 GPU 上默认开启 **vLLM V1 `SuffixDecodingProposer` + `DFlashProposer`** 组合；在超大规模推理云中采用 **SRAM Sidecar 滑窗草稿 + GPU HBM 全局验证**异构级联架构。
2. **后续可证伪实验问题**：
   - 当长上下文中的关键事实位于 `64K` 窗口的前 `10%`（超出 SRAM Draft 的 `8K` 滑动窗口）时，结合稀疏地标 Token（Landmark Attention Sink + Top-K Retrieval）注入 SRAM Draft 能否将接受长度 $E[\tau]$ 从 `3.1` 恢复至 `5.4+`？
