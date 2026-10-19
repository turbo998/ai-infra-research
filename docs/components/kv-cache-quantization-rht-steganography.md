# 超长上下文 KV Cache 量化架构：`FP8`、`INT4 + Walsh-Hadamard 旋转 + 零点隐写术`、`KIVI` 与 `NVFP4`

> **关联研究问题**：`kv-cache-quantization-fp8-nvfp4-kivi-outlier-rotation`（Priority 26）
> **关联实验**：[`experiments/kv_cache_quant_kivi_nvfp4/run.py`](../../experiments/kv_cache_quant_kivi_nvfp4/run.py)
> **关联单测**：[`tests/test_kv_cache_quant_kivi_nvfp4.py`](../../tests/test_kv_cache_quant_kivi_nvfp4.py)
> **证据登记**：`ev-vllm-int4-rht-steganography-kv-2026`、`ev-vllm-fp8-per-token-head-kv-2026`、`ev-exp-kv-cache-quant-kivi-nvfp4-2026`

---

## 1. 为什么 Key Cache 与 Value Cache 具有截然不同的量化敏感性？

在 `128K–1M` 超长上下文推理中，KV Cache 不仅占据了 GPU HBM 容量的主体，更决定了 Decode 阶段每个生成步的 HBM 读取字节数。然而，直接将 KV Cache 压至低比特（如 `INT4` 或静态 `FP8`）会遭遇严重的注意力精度崩塌：
1. **Key Cache 的固定通道离群值（Persistent Channel Outliers）**：在 RoPE 旋转位置编码前后，Key 张量的特定特征维度（例如 $d \in \{3, 19, 47, \dots\}$）在所有 Token 上都呈现比均值高 **`15x–30x`** 的幅值。若对每个 Token 沿 Head Dimension ($d=128$) 直接做对称/非对称 4-bit 量化，这极少数离群通道会撑大整個 Head 的量化步长 `scale`，导致其余 `95%` 正常通道的有效分辨率退化为 `1–2 bits`，引发 `Softmax(Q K^T / sqrt(d))` 分布严重偏移。
2. **Value Cache 的逐 Token 局部性**：Value 张量在 Softmax 概率加权求和是在 Token 维度上跨序列聚合的，按 Token 量化（Per-Token Quantization）能够自然对齐每个 Token 的激活幅值。
3. **量化元数据（Scale + Zero-Point）的带宽与对齐膨胀**：若为每组 16/32 个元素单独存储 FP16 `scale` 与 `zero_point`，不仅破坏 PagedAttention 连续槽位对齐，还会使每元素有效位数从 `4.0 bits` 膨胀至 `5.0–6.0 bits`，并增加 Decode Kernel 的二次全局内存加载延迟。

---

## 2. 一手源码解剖：`vllm-project/vllm` V1 KV Cache 量化内核

本研究直接核验了 `vllm-project/vllm`（commit `afe2183f2bab35988de0e6a506a5694b3cc1ef34`）中的三个核心源码文件：
- `vllm/v1/kv_cache_interface.py`（blob `c95439abf1eeacce7f9c88cb9a73cafd0847624a`）
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`（blob `47261bbc739a18cc6ae0df923ac2d1f299077ee1`）
- `vllm/v1/attention/ops/int4_per_token_head.py`（blob `3567f49d952c42441adbb15c126461ea7381fbfb`）

### 2.1 `KVQuantMode` 三种量化粒度与 Scale Cache 拓扑

在 `vllm/v1/kv_cache_interface.py`（lines 27–33）与 `triton_reshape_and_cache_flash.py`（lines 142–245）中，vLLM V1 明确区分了三种量化模式：
- **`KVQuantMode.PER_TENSOR`**：全层共享标量 `k_scale, v_scale`（`8.00 bits/elem`），无需额外 Scale 缓冲区，但长序列出现罕见大激活 Token 时全序列分辨率受损；
- **`KVQuantMode.PER_TOKEN_HEAD` (`FP8`)**：在 `_reshape_cache_per_token_head` Triton Kernel 中，对每个 `(token, kv_head)` 在寄存器内沿 $d=128$ 归约 `absmax = tl.max(tl.abs(k_vals))`，动态计算 `k_scale = tl.maximum(absmax / QUANT_MAX, 1e-6)` 并写入形状为 `(num_blocks, block_size, num_kv_heads)` 的 FP32 `k_scale_cache / v_scale_cache`（每 128 个 FP8 元素附带 4B FP32 scale，有效位宽 **`8.25 bits/elem`**）。

### 2.2 `INT4_PER_TOKEN_HEAD`：随机 Walsh-Hadamard 旋转（`single_rht`）与零点隐写术（Zero-Point Steganography）

在 `vllm/v1/attention/ops/int4_per_token_head.py` 中，vLLM V1 实现了一套极具巧思的 **`4.25 bits/elem`** 无额外零点缓冲区量化方案：

#### (1) 正交随机 Walsh-Hadamard 变换（`single_rht`，lines 90–145 & 980–1060）
在将 K、V 写入 Paged KV Cache 之前，以及对 Query 发起 Attention 计算之前，内核对长度为 $d=128$ 的向量施加正交变换：
$$\tilde{\mathbf{x}} = \mathbf{H}_d \mathbf{D}_1 \mathbf{x}, \quad \text{其中 } \mathbf{D}_1 = \text{diag}(s_1, \dots, s_d),\; s_i \in \{-1, +1\} \text{（由种子 } \texttt{0x9E3779B9} \text{ 生成）}$$
由于 $(\mathbf{H}_d \mathbf{D}_1)^T (\mathbf{H}_d \mathbf{D}_1) = \mathbf{I}_d$，内积完全守恒：
$$\langle \tilde{\mathbf{q}}, \tilde{\mathbf{k}} \rangle = \mathbf{q}^T \mathbf{D}_1^T \mathbf{H}_d^T \mathbf{H}_d \mathbf{D}_1 \mathbf{k} = \langle \mathbf{q}, \mathbf{k} \rangle$$
但 $\mathbf{H}_d \mathbf{D}_1$ 将原本集中在单一通道的 $18\times$ 尖峰能量均匀扩散到全部 $d=128$ 个通道，使旋转后的无穷范数 $\lVert \tilde{\mathbf{k}} \rVert_\infty$ 下降约 **`3.2x–4.1x`**，将各通道分布转化为近似高斯分布，从而允许对 K 和 V 统一采用 cache-friendly 的 **Per-(Token, Head)** 非对称 INT4 量化。

#### (2) FP32 Scale 尾数零点隐写术（Zero-Point Steganography，lines 240–272 & 345–368）
非对称 INT4 量化同时需要每个 `(token, head)` 的 `scale`（FP32）和 `zero_point`（整数 $\in [0, 15]$）。为了**不增加任何额外的 `zp_cache` 显存张量或全局内存读取指令**，`_reshape_cache_int4_kernel` 将 4-bit `zero_point` 直接嵌入 `float32` `scale` 的最低 4 位尾数（Mantissa LSBs）中：

```python
# vllm/v1/attention/ops/int4_per_token_head.py (lines 251-269)
k_scale = tl.maximum((k_max - k_min) / 15.0, 1e-6)
k_zp_int = tl.clamp(tl.extra.cuda.libdevice.rint(-k_min / k_scale), 0.0, 15.0).to(tl.int32)

# Steganography: pack 4-bit zero-point into the lowest 4 mantissa bits of float32 k_scale
k_scale_bits = k_scale.to(tl.int32, bitcast=True)
k_scale_packed = ((k_scale_bits & -16) | (k_zp_int & 0xF)).to(tl.float32, bitcast=True)
tl.store(k_scale_ptr + scale_offset, k_scale_packed)
```
在 Decode Attention 读取时，只需一次 4-Byte `tl.load` 即可同时恢复 `zp = scale_bits & 0xF` 与 `scale = (scale_bits & -16).to(tl.float32, bitcast=True)`。因为 IEEE-754 `float32` 拥有 23 位尾数，抹除最低 4 位尾数引入的相对尺度误差上限仅为 $2^{-19} \approx 1.9 \times 10^{-6}$（实测均值 **`1.37 ppm`**），对注意力输出的影响完全可忽略，却实现了 **零额外元数据空间与零额外 HBM 事务**！

---

## 3. 定量基准实验与全景对比（`experiments/kv_cache_quant_kivi_nvfp4/`）

我们在 [`experiments/kv_cache_quant_kivi_nvfp4/run.py`](../../experiments/kv_cache_quant_kivi_nvfp4/run.py) 中使用带真实固定通道离群值（4 个 Key 离群通道放大 `14x–18x`）的 GQA 注意力工作负载，系统评测了 7 种 KV Cache 量化方案的精度、隐写术误差与 **H200 (141 GiB, 4.8 TB/s)** 上的 Llama-3.1-70B 服务性能。

### 3.1 七种 KV Cache 格式的注意力保真度与 H200 容量/吞吐对比

| KV Cache 量化模式 | 有效位宽 (`bits/elem`) | 压缩比 (`vs BF16`) | 注意力权重 KL 散度 ($\downarrow$) | 输出 MSE ($\downarrow$) | 输出余弦相似度 ($\uparrow$) | H200 70B 最大缓存 Tokens | `BS=32 @ 32K` Decode TPOT (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bf16_baseline`** | `16.00` | `1.000x` | `0.000000` | `0.0000000` | `1.000000` | `238.0K` | `107.75` |
| **`fp8_per_tensor`**（静态全层 Scale） | `8.00` | `2.000x` | `0.007525` | `0.0000483` | `0.999850` | `476.1K` | `59.75` |
| **`fp8_per_token_head`**（动态逐 Token/Head Scale） | `8.25` | `1.939x` | **`0.000020`** | **`0.0000007`** | **`0.999998`** | `461.6K` | `61.18` |
| **`int4_per_token_head_no_rht`**（无旋转直接 INT4） | `4.25` | `3.765x` | `0.016167` | `0.0002072` | `0.999357` | `896.1K` | `41.35` |
| **`int4_per_token_head_rht_stego`**（vLLM RHT + 隐写术） | `4.25` | `3.765x` | **`0.000993`** | **`0.0000697`** | **`0.999778`** | **`896.1K`** | **`41.35`** |
| **`kivi_asymmetric_2bit_g32`**（K Per-Channel + V Per-Token + `R=32` 残差窗） | `2.85` | `5.611x` | `0.004057` | `0.0002840` | `0.999117` | **`1,335.5K`** | **`35.42`** |
| **`nvfp4_e2m1_block16_rht`**（Blackwell Block-16 `E2M1 + UE4M3` + RHT） | `4.50` | `3.556x` | **`0.000638`** | **`0.0000441`** | **`0.999859`** | `846.3K` | `42.20` |

### 3.2 核心定量结论与架构师选型指南

1. **动态 `FP8_PER_TOKEN_HEAD` 比静态 `FP8_PER_TENSOR` 降低 `376x` 注意力 KL 散度**：
   仅付出 `0.25 bits/elem`（`+3.1%` 显存）的 FP32 Scale 开销，动态 `FP8_PER_TOKEN_HEAD` 就将注意力 KL 散度从 `0.007525` 压至 `0.000020`（输出余弦相似度 `0.999998`），是在 Hopper / Blackwell 上实现无损 `~2x` KV 压缩的首选生产方案。
2. **Walsh-Hadamard 旋转 (`RHT`) + 零点隐写术是 `INT4` KV Cache 可用性的分水岭**：
   - 未旋转的 `INT4_PER_TOKEN_HEAD` 因 Key 通道离群值导致 KL 散度劣化至 `0.016167`；
   - 引入 `single_rht` 正交旋转与 FP32 尾数零点隐写术后，在保持 **`4.25 bits/elem`（`3.765x` 压缩比）** 不变的前提下，注意力 KL 散度骤降 **`16.3x`**（至 `0.000993`），输出 MSE 下降 **`2.97x`**，且隐写术对 Scale 的平均相对扰动仅为 **`1.37 ppm`**，将单卡 H200 的 Llama-3.1-70B 上下文容量从 `238.0K` 推升至 **`896.1K` tokens**，并将 `BS=32 @ 32K` Decode TPOT 从 `107.75 ms` 缩短至 **`41.35 ms`（提速 `2.61x`）**。
3. **Blackwell `NVFP4 (Block-16 E2M1 + RHT)` vs `KIVI 2-bit`**：
   - 在 Blackwell SM100 架构上，`NVFP4`（每 16 个元素共享 1B `FP8 E4M3` Scale 配合 RHT）以 `4.50 bits/elem` 取得了所有 sub-5-bit 方案中最低的 KL 散度（**`0.000638`**）与 MSE（`4.41e-5`）；
   - 当极长上下文（如 `> 1M` Agentic Memory）要求突破单卡百万 Token 容量时，`KIVI` 风格的非对称 2-bit（Key 按通道量化、Value 按 Token 量化并保留最近 `R=32` 个 Token 的 FP16 滑动残差窗）以 **`2.85 bits/elem`（`5.61x` 压缩）** 将单卡 H200 容量扩展至 **`1.335M` tokens**。
