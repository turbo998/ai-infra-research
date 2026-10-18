# NCCL `ncclCommRegister` / `ncclCommWindowRegister` 零拷贝用户缓冲区注册、CUMEM Fabric Handles 与免 SM 抢占集合通信

- **覆盖问题 ID**：`nccl-symmetric-memory-nvls-sharp-zero-copy`（`priority: 18`）
- **关联证据 ID**：`ev-nccl-user-buffer-reg-2026`、`ev-nccl-cumem-nvls-2026`、`ev-nccl-zero-copy-sim-2026`
- **关联离线实验**：`experiments/nccl_zero_copy_registration/run.py`（`results.json`）
- **上游源码核验基线**：`NVIDIA/nccl` (`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`)
  - `src/register/register.cc` (`blob 8ea5b4fa0601970883b42fc8914cf8ec6ca1d92e`)
  - `src/register/coll_reg.cc` (`blob d41a33894b179d7f1a984f29c223be34a2affd80`)
  - `src/allocator.cc` (`blob 417dd64bc8259b41848fea7b58ee77f1c6da0c84`)
  - `src/transport/nvls.cc` (`blob 155f6adf30223e19fb52fa4ed1c30b2014f27c44`)

---

## 1. 架构痛点：为什么默认未注册 NCCL 通信会拖慢并发 GEMM 达 `16%–47%`？

在现代训练与推理引擎（如 DeepSeek `DualPipe`、`FSDP2` 异步预取、分离式推理或 `Async-TP`）中，计算流（Compute Stream）与通信流（NCCL Stream）高度并发执行。然而，传统未注册缓冲区（Unregistered User Buffer）下的 NCCL 集合通信面临两大隐性代价：

1. **高额的内部中转缓冲池（Staging FIFO Buffers）显存税**：
   - 未注册用户张量无法被远端 GPU 或 RDMA 网卡直接 DMA 读写。NCCL 必须在每个 Communicator 的每个 Channel（通常 `16–32` 个 Channel）上为 `Simple`（默认 `4 MiB–8 MiB/peer`）、`LL128` 与 `LL` 协议预分配固定内部 FIFO 缓冲池。在 5D 并行同时激活 `TP`、`CP`、`EP`、`PP`、`FSDP` 五组 Communicator 时，仅 NCCL 内部缓冲池就吞噬 **`1.4 GiB–2.5 GiB` / GPU** 的宝贵 HBM。
2. **通信 Kernel 抢占 GPU SM 引发的 Wave Quantization（尾波膨胀）灾难**：
   - 为了将数据从 `user_sendbuff` 搬运到内部 `staging_FIFO` 并轮询协议标志位，NCCL 会启动 `8–32` 个常驻 CTA（每个 CTA 独占 1 个或半个 GPU SM）。
   - 以 H100 SXM5（`132` 个 SM）运行 `1,024` 个 CTA 的主力 GEMM 为例：无通信干扰时仅需 $\lceil 1024 / 132 \rceil = 8$ 个 Wave（`1.720 ms`）；一旦并发 NCCL 抢占 `16` 个或 `24` 个 SM，可用计算 SM 降至 `116` 或 `108` 个，导致 GEMM 尾波膨胀至 $\lceil 1024 / 116 \rceil = 9$ 个或 $\lceil 1024 / 108 \rceil = 10$ 个 Wave，叠加 FIFO 双重读写对 HBM 带宽的争抢，使并发 GEMM 耗时劣化 **`16.46%–31.60%`**（`32` CTAs 时高达 **`47.18%`**）。

---

## 2. 一手源码走读：NCCL `CUMEM` 分配器、注册缓存与 NVLS 零拷贝直通

### 2.1 `ncclMemAlloc` 与 `CU_MEM_HANDLE_TYPE_FABRIC`（`src/allocator.cc`）

在 `src/allocator.cc` (`blob 417dd64bc8259b41848fea7b58ee77f1c6da0c84`) 中，NCCL 实现了基于 CUDA Virtual Memory Management (`cuMem*`) 的物理内存池：

- **Fabric 句柄优先分配**：`ncclMemAlloc` 调用 `cuMemCreate` 分配物理显存粒度块，优先请求 `CU_MEM_HANDLE_TYPE_FABRIC`（支持跨节点 NVLink 域与 IMEX 导出，若不支持则回退至 `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`），随后通过 `cuMemAddressReserve`、`cuMemMap` 与 `cuMemSetAccess` 映射到虚拟地址空间。
- **奇偶切割区间树子分配器（`ncclSpace` & `ncclShadowPool`）**：为了避免频繁调用高开销的 `cuMemCreate`，`ncclSpace` 使用有序切割点数组 `cuts[]`（满足不变量 `isFull(i) = (i%2 != ncuts%2)`）在大型 Slab 上管理子块分配，并通过 `ncclShadowPool` 维护页对齐的设备内存区间映射。

### 2.2 用户缓冲区注册状态机（`src/register/register.cc`）

在 `src/register/register.cc` (`blob 8ea5b4fa0601970883b42fc8914cf8ec6ca1d92e`) 中，`ncclCommRegister` / `ncclRegister` 执行严格的物理段核验与多传输层绑定：

1. **页对齐与物理段扫描**（L56–116）：
   - 将用户传入的 `[addr, addr + size]` 向外对齐到 `cache->pageSize`：`begAddr = addr & -cache->pageSize`，`endAddr = (addr + size + cache->pageSize - 1) & -cache->pageSize`。
   - 当 `ncclCuMemEnable()` 开启时，循环调用 `cuMemGetAddressRange(&baseSend, &baseSendSize, currentAddr)` 遍历用户缓冲区覆盖的所有底层 `cuMem` 物理段，并检查 `CU_POINTER_ATTRIBUTE_MEMORY_TYPE`；若发现任何段位于主机内存（`hasSysmemSegment`），则立即跳过零拷贝注册以保安全。
   - 若未开启 `cuMem`，则检查 `CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE`。
2. **有序槽位缓存与四类传输状态位**（L118–205）：
   - 在 `comm->regCache.slots[]` 中按 `[begAddr, endAddr]` 二分查找或插入 `struct ncclReg` 记录（支持引用计数 `refCount`）；
   - `reg->state` 位掩码分别追踪四条零拷贝通路：`NET_REG_COMPLETE`（RDMA NIC `ibv_reg_mr`/`ibv_reg_dmabuf_mr`）、`NVLS_REG_COMPLETE`（NVSwitch 多播地址绑定）、`COLLNET_REG_COMPLETE`（IB SHARP 树注册）与 `IPC_REG_COMPLETE`（P2P IPC 映射）。

### 2.3 NVLS 多播零拷贝绑定与 Channel CTA 压缩（`coll_reg.cc` & `nvls.cc`）

在 `src/register/coll_reg.cc` (`blob d41a33894b179d7f1a984f29c223be34a2affd80`) 与 `src/transport/nvls.cc` (`blob 155f6adf30223e19fb52fa4ed1c30b2014f27c44`) 中：

- **本地注册与 CUDA Graph 自动注册**（`nvls.cc` L607–731）：
  - `ncclNvlsLocalRegisterBuffer` 调用 `nvlsFindLocalReg` 在 `regCache` 中查找 `sendbuff` 与 `recvbuff`（检查 `numSegments > 1 && !ncclParamMultiSegmentRegister()`），随后调用 `ncclNvlsUbRegister` 将用户缓冲区物理句柄直接挂入 NVSwitch 多播组（Multicast Group），返回 `*outRegBufUsed = 1` 及多播虚拟指针 `outRegBufSend` / `outRegBufRecv`。
  - 若处于 CUDA Graph 捕获期，`ncclNvlsGraphRegisterBuffer`（L684–731）自动触发 `nvlsGraphReg -> ncclCommGraphRegister` 并在图销毁时回调 `ncclCommGraphDeregister`。
- **零拷贝下的 Channel 数量自适应裁剪**（`nvls.cc` L733–789）：
  - `ncclNvlsRegResourcesQuery(comm, func, &recChannels)` 根据 `compCap`（Hopper `<100` vs Blackwell `>=100`）与 `comm->nvlsResources->nHeads` 重新计算所需 Channel 数。由于省去了内部 FIFO 搬运，NVSwitch 硬件直接对注册后的多播地址执行 `multimem.ld_reduce` / `multimem.st.bulk`，搭配 `ncclCommWindowRegister` 或设置 `comm->config.maxCTAs = 2` 即可在 `1–2` 个 CTA 下跑满 NVLink 带宽。

---

## 3. 定量实验与架构对比（`experiments/nccl_zero_copy_registration`）

运行 `python experiments/nccl_zero_copy_registration/run.py --check` 得到如下确定性量化结果：

### 3.1 通信 CTA 抢占对并发 GEMM 的 Wave Quantization 冲击（H100 `132 SMs`, `1024-CTA` GEMM）

| 通信模式 | NCCL 占用 CTA 数 | 剩余计算 SM 数 | GEMM 所需 Wave 数 | 并发 GEMM 耗时 (`ms`) | GEMM 性能劣化幅度 |
|---|---:|---:|---:|---:|---:|
| `isolated_gemm_no_comm`（纯计算基线） | 0 | 132 | 8 | **1.720** | **0.00%** |
| **`zero_copy_nvls_window_reg`（零拷贝注册 + NVLS）** | **2** | **130** | **8** | **1.720** | **0.00%** |
| `unreg_ring_8_channels`（未注册 8-CTA Ring） | 8 | 124 | 9 | 1.969 | +14.48% |
| `unreg_ring_16_channels`（未注册 16-CTA Ring） | 16 | 116 | 9 | 2.003 | **+16.46%** |
| `unreg_tree_nvls_24_channels`（未注册 24-CTA 多轨） | 24 | 108 | 10 | 2.264 | **+31.60%** |
| `unreg_max_32_channels`（未注册 32-CTA 满通道） | 32 | 100 | 11 | 2.531 | **+47.18%** |

### 3.2 多维并行下 NCCL 内部缓冲池 HBM 节省（每 GPU）

| 集群并行拓扑 | 活跃 Communicator 数 | 每 Comm Channel 数 | 未注册 FIFO 缓冲池占用 | 零拷贝控制元数据 | 每 GPU 净省 HBM |
|---|---:|---:|---:|---:|---:|
| 单机 `TP=8 + DP=8` | 2 | 16 | `640.0 MiB` (`0.625 GiB`) | `64.0 MiB` | **`576.0 MiB` (`0.562 GiB`)** |
| 3D 并行 `TP=8 + PP=4 + FSDP=32` | 3 | 24 | `1440.0 MiB` (`1.406 GiB`) | `96.0 MiB` | **`1344.0 MiB` (`1.312 GiB`)** |
| 5D MoE `TP=8 + CP=2 + EP=32 + PP=4 + FSDP=16` | 5 | 24 | `2520.0 MiB` (`2.461 GiB`) | `160.0 MiB` | **`2360.0 MiB` (`2.305 GiB`)** |

### 3.3 不同消息尺寸下的集合通信 P99 延迟加速比

| 消息尺寸 (`MiB`) | 未注册缓冲区 P99 (`us`) | 零拷贝 NVLS 注册 P99 (`us`) | P99 延迟加速比 |
|---:|---:|---:|---:|
| **0.25 MiB** (`256 KiB`) | 35.54 | **9.49** | **3.74x** |
| **1.00 MiB** | 39.91 | **11.40** | **3.50x** |
| **4.00 MiB** | 57.40 | **19.04** | **3.02x** |
| **16.00 MiB** | 127.36 | **49.57** | **2.57x** |
| **64.00 MiB** | 358.45 | **168.55** | **2.13x** |
| **256.00 MiB** | 1343.81 | **648.11** | **2.07x** |

---

## 4. 解决方案架构师配置契约与可证伪问题

1. **生产环境启用零拷贝用户缓冲区注册的三要素**：
   - 张量必须由 `ncclMemAlloc` 或 PyTorch `MemPool(allocator=_nccl_allocator)` / `expandable_segments:True` (`cuMemCreate` 分配器) 分配，避免普通 `cudaMalloc` 跨多个非连续物理段（否则需显式开启 `NCCL_MULTI_SEGMENT_REGISTER=1`）；
   - 开启 `NCCL_NVLS_ENABLE=1` 与 `NCCL_LOCAL_REGISTER=1`（或在 CUDA Graph 捕获下自动激活 `ncclNvlsGraphRegisterBuffer`）；
   - 通过 `ncclConfig_t.maxCTAs` 或 `TORCH_NCCL_HIGH_PRIORITY` 限制并发通信流 CTA 数，确保计算 GEMM 的 `132 / 148` 个 SM 不跨越 Wave Quantization 台阶。
2. **后续可证伪实验问题**：
   - 当用户开启 PyTorch `expandable_segments:True` 且单个大张量由 `>8` 个 `2 MiB` 物理 `cuMem` 段拼接而成时，`NCCL_MULTI_SEGMENT_REGISTER=1` 在 NVSwitch TLB 与 ConnectX-7 MTT（Memory Translation Table）中会产生多少页表未命中（MTT Cache Miss）延迟？
