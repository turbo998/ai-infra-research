# DeepSeek 3FS (`hf3fs_usrbio`)、GPUDirect Storage (`cuFile`) 与多级 KV Cache / Checkpoint I/O 管线

## 1. 核心架构问题与结论（Executive Summary）

在万亿参数 MoE 训练的 TB 级周期性 Checkpoint 落盘与长上下文/PD 分离/Agentic 推理的百 GB 级历史 KV Cache 卸载（Tiered KV Offloading）中，传统存储路径存在两道致命的软件与总线墙：
1. **POSIX Page Cache 双缓冲与 PCIe Root Complex Bounce 税**：标准 `pread`/`pwrite` 会先将 NVMe/RDMA 数据拷入内核 Page Cache，再 `memcpy` 到用户态缓冲区，最后通过 `cudaMemcpyAsync` 经由 CPU Root Complex 拷入 GPU HBM。单次 KV 读取在主机 DDR 内存总线上产生 **`2–3` 次冗余往返**，不仅将 `8 × PCIe Gen5 NVMe`（物理线速 `56.0 GB/s`）的有效 GPU 吞吐压低至 **`32.0 GB/s`**，还占用了 **`53.3%`** 的主机内存带宽与高达 **`88.0%`** 的 PCIe Root Complex 事务开销，直接引发同节点 Decode 线程的 TPOT 抖动。
2. **Linux FUSE 内核自旋锁的 400K IOPS 墙**：在容器化环境中直接挂载 FUSE 文件系统访问分布式存储时，所有 I/O 请求必须跨越用户态-内核态边界并争抢内核 FUSE 请求队列自旋锁（Spinlock）。我们对 `deepseek-ai/3FS`（commit `22fca04564c7cc230fd8b9523b8b92864e1dad47`）源码与设计文档的核验表明，原生 FUSE 在 `4 KiB` 随机读下被死死卡在 **`~400K IOPS`**，且产生高达 `60%` 的内核态 CPU 开销。
3. **GPUDirect Storage (`cuFile` + `nvidia-fs`) 与 DeepSeek 3FS `USRBIO` 的两条零拷贝破局路径**：
   - **本地/NVMe-oF BAR1 P2P DMA（`NVIDIA/gds-nvidia-fs`）**：通过内核 `nvfs-dma.c` 拦截 `blk_rq_dma_map_iter_start`，调用 `nvidia_p2p_get_pages` / `nvidia_p2p_dma_map_pages` 将 GPU BAR1 物理页直接暴露给同 PCIe Switch 下的 NVMe SSD 或 RDMA 网卡，绕过 CPU 内存实现 **`54.4 GB/s`** 有效吞吐（PCIe RC 负载降至 `5.0%`）；但必须确保 PCIe 拓扑不跨 Root Port 且关闭 PCIe Bridge **ACS（Access Control Services）重定向**。
   - **DeepSeek 3FS `hf3fs_usrbio` 共享内存环 + RDMA 零拷贝（`deepseek-ai/3FS` & `LMCache`）**：客户端通过 `hf3fs_iovcreate` 在 `/dev/shm` 创建共享内存并在 `<mount>/3fs-virt/iovs/` 下注册带编码参数的虚拟符号链接，由本地 `fuse.hf3fs` 守护进程代管 InfiniBand Memory Region (`ibv_reg_mr`) 并通过无锁 `hf3fs_ior` 提交环绕过内核 FUSE。需要注意：**`LMCache` 当前的 `Hf3fsConnector`（`csrc/storage_backends/hf3fs/connector.cpp`）仍在调用者缓冲区与 `iov.base` 之间执行了一次主机侧 `memcpy`（实测模型上限 `48.0 GB/s`）**；只有将 CUDA 锁页内存或 GPU BAR1 直接注册进 `hf3fs_iov`（GPUDirect RDMA 零拷贝路径），才能跑满双口 `400GbE/NDR` 的 **`90.0 GB/s`** 聚合吞吐，将 `100 GiB` KV 恢复 P99 延迟从 `5.536 s` 压降至 **`1.265 s`（提速 `4.38x`）**，将 `1 TiB` Checkpoint 写入 P99 从 `56.694 s` 压降至 **`17.986 s`（提速 `3.15x`）**。

---

## 2. 一手源码深度剖析（`deepseek-ai/3FS`、`LMCache` 与 `NVIDIA/gds-nvidia-fs`）

### 2.1 `deepseek-ai/3FS`：`hf3fs_usrbio` 虚拟符号链接注册协议与 CRAQ + BIBD 链式复制

在 `deepseek-ai/3FS`（commit `22fca04564c7cc230fd8b9523b8b92864e1dad47`，`2026-05-07`）的 `src/lib/api/hf3fs_usrbio.h`（blob `c8c237e58c16021227ae2cfdf1c938dccd50b852`）、`src/lib/api/UsrbIo.cc`（blob `dbcd3ebca72a057d4c7affa082c8cdacca745dc3`）、`docs/design_notes.md`（blob `e652b8771719454c7d40b0b36cd64718c2d92dee`）与 `src/storage/service/StorageOperator.cc`（blob `04dcd5b24cafb5d3772cdb213a29e560ed2552d6`）中，3FS 展示了专为 AI 集群设计的用户态零拷贝分布式文件系统内核：

1. **`/dev/shm` + `<mount>/3fs-virt/iovs/` 符号链接控制通道（`UsrbIo.cc`）**：
   - 校验挂载点魔数 `HF3FS_SUPER_MAGIC = 0x8f3f5fff`；
   - `hf3fs_iovcreate_general` 先在 `/dev/shm/hf3fs-iov-<uuid>` 分配大块共享内存（`struct hf3fs_iov`），随后在 FUSE 挂载点的虚拟目录 `<mount_point>/3fs-virt/iovs/` 下执行 `symlink()`，链接名直接将 I/O 环参数编码为字符串：
     ```text
     <hex_id>.b<block_size>.<r|w><io_depth>.p<h|l>.t<timeout>.f<flags>
     ```
   - 驻留在本机的 `fuse.hf3fs` 守护进程拦截该 `symlink` 系统调用，直接打开对应的 `/dev/shm` 共享内存段，完成 InfiniBand `ibv_reg_mr` 物理页钉住（Pinning）与轮询环（`struct hf3fs_ior`）映射。此后应用调用 `hf3fs_prep_io` + `hf3fs_submit_ios` 完全在用户态共享内存中投递 I/O 描述符，**零内核 FUSE 上下文切换**。
   - **线程安全契约（`UsrbIo.md`）**：单个 `hf3fs_ior` 环上的 `hf3fs_prep_io` **不是线程安全的**；多线程并发卸载 KV Cache 时，必须为每个工作线程分配独立的 `hf3fs_ior` 或在应用层加锁。
2. **CRAQ（支持脏读查询的链式复制）+ BIBD（均衡不完全区组设计）恢复（`design_notes.md` & `StorageOperator.cc`）**：
   - **Write-All, Read-Any**：写请求沿复制链顺序传递，但**数据体并非通过 RPC 推送，而是由下游 Storage Target 发起 RDMA Read 主动从客户端/上游拉取（Pull）**，避免接收端缓冲区溢出；当尾节点（Tail）确认写入后，客户端即可从链上**任意一个副本（Read-Any）**就近读取，最大化利用全网 NVMe 读带宽。
   - **BIBD 链打散防止热点雪崩**：若按传统固定三元组建链，当某颗 SSD 损坏离线时，其原本承担的读流量会 100% 压向同链的剩余两个副本，形成读放大瓶颈。3FS 利用组合数学中的 **Balanced Incomplete Block Design (BIBD)** 构造多条交叉复制链，使得任意单盘故障时，其读负载被均匀分摊到全集群所有存活节点上。
   - **FoundationDB SSI 元数据与动态条带（Stripe）**：元数据服务完全无状态，底层依托 FoundationDB 的可串行化快照隔离（SSI）事务；Inode 键前缀为 `"INOD"` + 小端序 Inode ID，目录项为 `"DENT"` + 父 Inode ID + 文件名；大文件默认切分为 `200` 个 Stripe Targets，并从 `16` 开始动态倍增 hint 以兼顾小文件元数据开销与大文件并发吞吐。

### 2.2 `LMCache/LMCache`：`Hf3fsConnector` 与 `CuFileBackend` 的实现对比

在 `LMCache/LMCache`（commit `663757e38a1cc98fa1f6c0d76af8965c4fbb53e7`，`2026-10-18`）中，我们对比了其 3FS 后端与 GDS 后端的真实代码路径：
1. **`csrc/storage_backends/hf3fs/connector.cpp`（blob `c895cc56742a212a48d943f69d04839cb6fb7acc`）**：
   - `Hf3fsConnector` 调用 `hf3fs_iorcreate4` 与 `hf3fs_iovcreate` 初始化环；
   - **原子写入契约**：写入 KV 块时，先以 `O_WRONLY | O_CREAT | O_EXCL` 打开临时文件 `file_path + "." + random()`，通过 `hf3fs_submit_ios` 写完并 `fsync` 后，再调用 `std::filesystem::rename(temp_file_path, file_path)` 原子发布，防止其他推理实例读到半截 KV 块；
   - **64-bit 哈希目录分片**：截取 `chunk_hash` 末尾 16 个十六进制字符（64 bits）对 `base_paths_` 取模分片；
   - **隐藏的 Host `memcpy` 瓶颈**：在 `batch_read` / `batch_write` 循环内部，代码执行了 `std::memcpy(static_cast<char*>(buf) + total_read, iov.base, sub_len)` 将数据在调用者传入的 PyTorch Tensor 缓冲区与 `/dev/shm` 的 `iov.base` 之间拷贝了一次！这意味着若不将 `iov.base` 直接绑定为推理引擎的 Pinned/BAR1 内存池，主机 DDR 带宽仍会被额外消耗一次。
2. **`lmcache/v1/gpu_connector/gds_backends/cufile.py`（blob `eb2089ca2aa1de78820e563b52b50bafb33b3908`）**：
   - 直接通过 `ctypes` 绑定 `libcufile.so` 的 `cuFileBufRegister`、`cuFileStreamRegister`（标志位 `_STREAM_REGISTER_FLAGS = 0x7`）以及异步 `cuFileReadAsync` / `cuFileWriteAsync`，使 KV Cache 块直接沿 CUDA Stream 从 NVMe/文件系统 DMA 进出 GPU 显存。

### 2.3 `NVIDIA/gds-nvidia-fs`：BAR1 P2P 映射与 PCIe ACS 拓扑惩罚矩阵

在 `NVIDIA/gds-nvidia-fs`（commit `52f3e7eab633103478e22cc2c2ac1c6ec368eea3`，`2026-09-17`）的 `src/nvfs-dma.c`（blob `a72bcc5b21309ba6a51b029bcc8e7f2fb8597015`）与 `src/nvfs-pci.c`（blob `f24ac0fc28f52b1b54856ac35f58a7b6560204c3`）中，内核驱动揭示了 GPUDirect Storage 达到物理线速的硬件先决条件：
1. **块层 DMA 拦截与 GPU 页钉住（`nvfs-dma.c` & `nvfs-p2p.h`）**：
   - `nvfs_blk_register_dma_ops()` 通过 `nvme_v2_register_nvfs_dma_ops` 接管 NVMe 驱动的 `blk_rq_dma_map_iter_start` / `blk_rq_dma_map_iter_next`；
   - 调用 NVIDIA 闭源驱动导出的 `nvidia_p2p_get_pages` 与 `nvidia_p2p_dma_map_pages` 获取 GPU HBM 对应 BAR1 窗口的物理总线地址，直接填入 NVMe PRP/SGL 描述符。
2. **PCIe 拓扑距离矩阵与 ACS 重定向陷阱（`nvfs-pci.c`）**：
   - `__nvfs_get_gpu2peer_distance` 计算每张 GPU 到每块 NVMe SSD / RDMA 网卡的拓扑距离：同 PCIe Switch 下距离最优；一旦跨越 CPU Root Port，即被打上 `BASE_PCI_DISTANCE_CROSSRP` 惩罚标签；
   - 驱动主动扫描整条 PCIe 路径上的桥设备扩展能力寄存器（`PCI_EXT_CAP_ID_ACS`，检查 `ctrl & 0x7f`）：**若主板 BIOS 或虚拟化层在 PCIe Switch 上开启了 ACS P2P Request Redirect，所有原本应在 PCIe Switch 内部闭环转发的 GPU-NVMe P2P TLP 报文都会被强制上送到 CPU Root Complex 绕一圈，导致 GDS 吞吐腰斩并挤爆 Root Port**。

---

## 3. 原创 CPU 离线实验结果（`experiments/storage_gds_kv_checkpoint/`）

运行 `python experiments/storage_gds_kv_checkpoint/run.py --check` 对 5 种存储 I/O 数据通路在 **`100 GiB` 长上下文 KV Cache 突发恢复** 与 **`1 TiB` 训练 Checkpoint 异步刷盘** 下的表现进行定量对比：

### 3.1 五种存储 I/O 数据通路吞吐、延迟与总线争用对比表

| 存储 I/O 路径 (`id`) | 有效 GPU 读吞吐 (`GB/s`) | 有效 GPU 写吞吐 (`GB/s`) | 主机 DDR 往返拷贝次数 | 主机 DDR 带宽占用率 (`%`) | PCIe Root Complex 负载 (`%`) | `100 GiB` KV 恢复 P99 (`s`) | `1 TiB` Checkpoint 写 P99 (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`posix_page_cache_buffered`**（传统 POSIX `pread` + Page Cache + `cudaMemcpy`） | `32.0 GB/s` | `22.0 GB/s` | `2.5x` | **`53.3%`** | **`88.0%`** | `5.536 s` | `56.694 s` |
| **`o_direct_pinned_bounce`**（`O_DIRECT` + 锁页内存 Bounce Buffer） | `44.0 GB/s` | `36.0 GB/s` | `1.0x` | `14.7%` | `68.8%` | `2.948 s` | `32.275 s` |
| **`gds_cufile_bar1_p2p`**（本地 NVMe GPUDirect Storage `cuFile` BAR1 P2P） | **`54.4 GB/s`** | **`46.8 GB/s`** | **`0.0x`** | **`0.0%`** | **`5.0%`** | `2.132 s` | `24.101 s` |
| **`hf3fs_usrbio_shm_memcpy`**（3FS `USRBIO` `/dev/shm` + `LMCache` Host `memcpy`） | `48.0 GB/s` | `40.0 GB/s` | `1.5x` | `24.0%` | `75.0%` | `2.702 s` | `28.545 s` |
| **`hf3fs_usrbio_gpudirect_p2p`**（3FS `USRBIO` + GPUDirect RDMA BAR1 零拷贝） | **`90.0 GB/s`** | **`64.0 GB/s`** | **`0.0x`** | **`0.0%`** | **`6.0%`** | **`1.265 s`** | **`17.986 s`** |

---

## 4. 面向解决方案架构师的部署清单与练习

### 4.1 生产环境 GDS + 3FS 核查清单（Checklist）
1. **PCIe ACS 检查**：在宿主机执行 `lspci -vvv | grep -i acsctl`，确认連接 GPU、NVMe 与 InfiniBand CX-7 的 PCIe Switch 下行端口 `SrcValid-`、`TransBlk-`、`ReqRedir-` 均已关闭；并检查 `/proc/driver/nvidia-fs/stats` 确认 `nvfs-pci` 未报告 `CROSSRP` 跨根端口降级。
2. **GPU BAR1 窗口容量校验**：通过 `nvidia-smi -q | grep -A 3 "BAR1 Memory Usage"` 确认启用了 Large BAR1（H100/B200 支持完整显存映射），防止大并发 `cuFileBufRegister` 耗尽 BAR1 触发回退。
3. **3FS `USRBIO` 内存池零拷贝改造**：在集成 `LMCache` 或自研 KV 连接器与 `3FS` 时，将 `hf3fs_iov` 的 `/dev/shm` 缓冲区直接注册为 `cudaHostRegister(..., cudaHostRegisterMapped | cudaHostRegisterIoMemory)`，消除 `connector.cpp` 中的二次 `std::memcpy`。

### 4.2 思考与计算练习
- **练习 1（PCIe Root Complex 争用导致的推理尾延迟放大）**：某 8-GPU H100 节点在运行 Decode 的同时，后台每秒通过 `posix_page_cache_buffered` 路径从本地 NVMe 加载 `16 GiB` 历史 KV Cache。请根据主机 DDR5 带宽与 PCIe Root Port 带宽计算该后台 I/O 占用了多少主机内存带宽？若改用 `gds_cufile_bar1_p2p`，为什么 Decode 线程的 Host-to-Device 控制投递抖动会显著消失？
- **练习 2（CRAQ + BIBD 故障读分摊计算）**：在由 `N = 16` 个存储节点组成的 3FS 池中，每个 Chunk 采用 3 副本 CRAQ 链。若不使用 BIBD 而采用固定分组 `(Node 1, 2, 3)`，当 `Node 1` 宕机时，`Node 2` 和 `Node 3` 的读负载增加多少百分比？采用 BIBD 均匀打散后，单节点故障对任意其他存活节点的读负载增量降至多少？
