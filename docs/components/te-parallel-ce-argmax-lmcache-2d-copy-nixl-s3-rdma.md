# TransformerEngine Fused Parallel Cross-Entropy Argmax, LMCache 2-D Block-Outermost Batch Copy & NIXL Multipart S3-over-RDMA

> **Language**: [中文](te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma.md) | [English](te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma.en.md)
> **Phase**: 65 (`priority: 74`) · **Track C**: Distributed Training, KV Cache Offloading & RDMA Object Storage
> **Verified Upstream Sources**: `NVIDIA/TransformerEngine@39a8b879df` (`#3671`), `NVIDIA/TransformerEngine@76ae4b0981` (`#3599`), `LMCache/LMCache@0d441be069` (`#5506`), `ai-dynamo/nixl@99d469be5d` (`#2406`), `ai-dynamo/nixl@14e329032b` (`#2172`), `kvcache-ai/Mooncake@d782a5998f` (`#4755`)

---

## 1. 核心架构问题与研究背景

在大词表分布式训练（TP Sharded Cross-Entropy + Accuracy 监控）、Block-Outermost 布局（如 DeepSeek-V4 `BLNHC`）的跨进程 KV Cache 直接传输，以及超大显存对象（>4 GiB）的 S3-over-RDMA 传输中，系统存在三类典型的 **二次扫描、描述符风暴与硬件注册边界瓶颈**：

1. **TransformerEngine `parallel_cross_entropy` 的二次 Logits HBM 扫描与 Triton `tl.max` NaN 语义漏洞 (`TransformerEngine@39a8b879df` & `76ae4b0981`)**：
   - 在 RLHF/SFT 训练需要同时计算交叉熵 Loss 和 Token 预测准确率（`argmax`）时，`parallel_cross_entropy` 的前向 Kernel 本已遍历了整行 Logits 以求取最大值 $m$。若调用方再执行一次 `torch.argmax(logits)`，必须将高达数百 MB 的词表 Logits 从 HBM 重新读入一遍。
   - 更隐蔽的是，Triton 的 `tl.max(..., return_indices=True)` **不是 NaN-safe 的**：由于所有针对 `NaN` 的比较均为 `False`，包含 `NaN` 的 Block（如 `[8, -10, NaN, -10]`）会返回既非最大值也非 `NaN` 的错误索引（如 `1`），违背了 `torch.argmax`“将 `NaN` 视为最大值且首个 `NaN`/首个并列最大值获胜”的严格语义。
   - 同时，在 Grouped MXFP8 变长尾维量化 (`76ae4b0981`) 中，多张量路径 (`is_single_tensor = false`) 在 `scales_rowwise` 指针已加上 `tensor_base / SCALE_DIM_X` 的情况下，又将 `tensor_base_for_scales` 设为 `tensor_base`，导致 Swizzled Scale 偏移被重复累加两次。
2. **LMCache 在 Block-Outermost KV 布局下的 1-D 拷贝描述符风暴 (`LMCache@0d441be069`)**：对于 vLLM 的 `BLNHC` 等 Block-Outermost 布局（例如 DeepSeek-V4 61 层 MLA），同一个 Block 在所有层的 Page 在物理显存中以固定步长（Pitch）紧密相邻排列。此前 Direct Copy 路径仍按 `(kv_plane, layer, block)` 逐层生成 1-D `cudaMemcpyBatchAsync` 条目，单次 32-Block Chunk 需提交 $61 \times 32 = 1952$ 个拷贝描述符。
3. **NIXL S3-over-RDMA 在单对象 $\ge 4\text{ GiB}$ 时的 `CU_OBJ_INVALID_VALUE` 失败 (`nixl@99d469be5d` & `14e329032b`)**：尽管理论上 CUDA 13.4 支持注册远超 4 GiB 的大显存 Arena（如 64 GiB），但底层 `cuObject` 单次请求签发 RDMA Token 的硬上限为 `CUOBJ_MAX_MEMORY_REG_SIZE` ($4\text{ GiB} - 64\text{ KiB} = 4,294,901,760\text{ B}$)。直接对 $\ge 4\text{ GiB}$ 对象发起单次 RDMA PUT/GET 会立即触发 `cuMemObjGetRDMAToken` 返回 `CU_OBJ_INVALID_VALUE`。

---

## 2. 源码级实现机制与公式推导

### 2.1 TransformerEngine NaN-Safe `parallel_cross_entropy(..., return_argmax=True)` (`39a8b879df`)

在 `transformer_engine/common/triton/cross_entropy.py` 中，新增了 `_is_nan(x)` (`return x != x`) 与 `_fold_block_argmax` 辅助函数，与 Loss 的在线 Softmax 统计量独立维护 `(run_max, run_idx, run_nan)`：

1. **Block 内 NaN 剥离与左优先 Tie-Break**：
   - 先用 `is_nan = (x != x)` 检测当前 Block 是否含 `NaN`，并计算首个 `NaN` 的列偏移 `first_nan = tl.min(tl.where(is_nan, offsets, n_cols))`。
   - 将 `NaN` 替换为 `-inf` 后调用 `tl.max(tl.where(is_nan, float("-inf"), x), axis=0, return_indices=True, return_indices_tie_break_left=True)`。
   - 若此前尚未遇到 `NaN` (`run_nan == 0`)：若当前 Block 含 `NaN`，则立即令 `run_idx = first_nan, run_nan = 1`；否则仅当 `block_max > run_max`（严格大于，保证首个并列最大值获胜）时才更新 `run_idx = i + block_idx, run_max = block_max`。
2. **TP 跨 Rank All-Gather 零额外通信轮次**：在词表张量并行（TP）下，每个 Rank 的局部 `argmax` 索引作为原始比特（1 列额外的 FP32）随现有的 Softmax 统计量 All-Gather 一并传输（若该 Rank 含 `NaN` 则共享 `NaN` 作为最大值），由持有全局最大值的首个 Rank 提供最终全局索引，Loss 与梯度保持 **逐比特不变**。
3. **Grouped MXFP8 Scale 基址修正 (`76ae4b0981`)**：
   ```cpp
   const size_t tensor_base_for_scales = is_single_tensor ? tensor_start_offset : 0;
   e8m0_t *const scales_rowwise =
       scales_rowwise_ptr + (is_single_tensor ? 0 : tensor_base / SCALE_DIM_X);
   ```
   消除了非单张量模式下的双倍 `tensor_base` 偏移错误。

### 2.2 LMCache Block-Outermost 2-D 批量拷贝 (`0d441be069`)

在 `csrc/cuda/mp_mem_kernels.cu` 中，`uniform_layer_pitch(addr, spec)` 判定一个 Kernel Group 的所有层是否可合并为单次 2-D 拷贝（`cudaMemcpy3DBatchAsync`）：
- 计算相邻层同一 Block 的字节间距 $\text{pitch} = \text{ptrs}[l+1] - \text{ptrs}[l]$。
- 当所有相邻层间距恒等于 $\text{pitch}$，且满足 **单 Block 跨层包络条件**：
  $$\text{pitch} \ge \text{block\_bytes} \ \land\ (N_L - 1) \cdot \text{pitch} + \text{block\_bytes} \le \text{block\_stride}$$
  说明该 KV Cache 为 Block-Outermost 布局（如 `BLNHC`），此时每个 `(kv_plane, block)` 只需提交 **1 个 2-D 拷贝条目**（$\text{width} = \text{block\_bytes}, \text{height} = N_L, \text{src/dstPitch} = \text{pitch}$），将描述符数量压缩 $N_L$ 倍；若为 Layer-Outermost 布局，则函数返回 `0` 并安全回退至逐层 1-D 拷贝。

### 2.3 NIXL 超大对象分段 S3-over-RDMA 与 GPUNetIO 双 SGE WRITE (`99d469be5d` & `14e329032b`)

在 `src/utils/object/rdma/rdmaRetry.cpp` 中，当对象大小 $S > \text{rdma\_max\_request\_size}$ ($4\text{ GiB} - 64\text{ KiB}$) 时：
- **Multipart RDMA PUT (`rdmaPutObject`)**：首先通过控制面 `cp.createMultipartUpload(ctx)` 创建分片上传，随后将缓冲区按最大分片长度 $P_{\max} = 4\text{ GiB} - 64\text{ KiB}$ 切分为 $K = \lceil S / P_{\max} \rceil$ 个分片。每个分片从同一块已注册的超大 VRAM Arena 中独立获取 `cuMemObjGetRDMAToken` 并执行 RDMA `UploadPart`；任一分片失败则自动调用 `cp.abortMultipartUpload(ctx)` 清理状态。
- **Ranged RDMA GET**：按 $\le 4\text{ GiB} - 64\text{ KiB}$ 的连续字节区间并发发起 Ranged GET 写入目标显存偏移。
- **GPUNetIO Peer VRAM 与 Two-SGE WRITE (`14e329032b`)**：支持注册非通信 GPU 持有的对端显存（Peer-owned VRAM），并将满足合并条件的两个相邻 WRITE 描述符直接打包进单个含 2 个 SGE（Scatter-Gather Elements）的 WQE 中，无需额外 Gather 缓冲区即可将 WQE 投递数减半。

---

## 3. 确定性实验验证结果 (`experiments/te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma`)

运行 `python experiments/te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma/run.py --check` 通过全部 8 项不变量校验：

- **TransformerEngine `parallel_cross_entropy` + MXFP8**:
  - 在 $T=4096, V=128,256, \text{TP}=8$ 下，每 Rank 省去二次读取 Logits 的 `131,334,144 B` (`125.25 MiB`) HBM 流量，而 All-Gather 仅增加 `16,384 B` (`16 KiB`)，净内存流量收益比达 **8016.0×**。
  - `[8.0, -10.0, NaN, -10.0]` 精确返回索引 `2`（首个 `NaN`）；`[3.5, 9.0, 9.0, 2.0, NaN, NaN]` 返回索引 `4`；跨 Rank 并列最大值 `[1.0, 7.5, 7.5, 4.0]` 返回首个全局索引 `1`。
  - Grouped MXFP8 修正后 Scale 偏移从错误的 `4096` 恢复为精确的 `2048`。
- **LMCache 2-D Direct Copy**:
  - 在 61 层、32 Blocks/Chunk (`block_bytes = 36,864 B`) 的 `BLNHC` 布局下，`uniform_layer_pitch` 检出 `36,864 B`，将 `cudaMemcpy3DBatchAsync` 描述符数量从 `1952` 降至 `32`（压缩 **61.0×**）；而在 Layer-Outermost 布局下返回 `0` 并回退至 1-D。
- **NIXL Multipart S3-over-RDMA & GPUNetIO**:
  - `9 GiB` 显存对象自动切分为 **3 个 RDMA 分片**（尾分片 `1,073,872,896 B`），`64 GiB` 显存对象自动切分为 **17 个 RDMA 分片**（尾分片 `1,048,576 B`），所有分片 100% 满足 $\le 4,294,901,760\text{ B}$ 的 `cuObject` 令牌限制；GPUNetIO 双 SGE 合并将 64 个描述符的 WQE 数量从 `64` 降至 `32`。

---

## 4. 面向解决方案架构师的设计建议与可证伪问题

### 架构师检查清单
1. **CUDA 13.4 超 32 GiB 显存注册系统参数**：在使用 NIXL S3-over-RDMA 传输 64 GiB 级单卡显存 Arena 时，除了启用本次的 Multipart RDMA 切分外，还必须在宿主机 `/etc/cufile.json` 中将 `max_device_pinned_mem_size_kb` 从默认的 `32 GiB` 调高，否则底层注册将提前失败。
2. **KV Cache 布局选型对跨进程传输的影响**：若推理引擎采用 `BLNHC`（Block-Outermost）布局，LMCache 可通过 2-D Pitch 拷贝将 DMA 描述符开销降低 $N_L$ 倍；但若运行在 ROCm/HIP 环境（`USE_ROCM`），当前 `LMC_HAS_2D_BATCH_MEMCPY` 宏默认关闭，仍走逐层 1-D 路径。

### 后续可证伪实验问题
- **Q1**: 当 NIXL S3-over-RDMA 传输恰好为 $4\text{ GiB}$ ($4,294,967,296\text{ B}$) 的整数倍对象时，由于 $4\text{ GiB} > 4\text{ GiB} - 64\text{ KiB}$，第 2 个分片仅包含 `65,536 B` (`64 KiB`)；将该切分阈值对齐为 `2 GiB` 或均分分片大小是否能改善尾分片的 RDMA 带宽利用率？
