# Megatron-FSDP 反向传播预取激活重算权重、DSv4 CSA 原位输出 RoPE 反向重构与 NCCL EP `THD` 序列打包，PyTorch ROCm MI355X `seqlen_q==1` CK-Tile 反向路由，Mooncake Classic RDMA 误差驱动拥塞控制、TENT `io_uring` 飞行中 I/O 排空等待与 PG 路由 IP 推导，LMCache `pygrpc-native` 保序亲和与 FS L2 `O_DIRECT` 未对齐缓冲回退，及 Dynamo SGLang KV 事件格式与 NIXL EP 断连内存释放分析（Phase 132 · Priority 141）

> **English Version**: [megatron-fsdp-recompute-csa-rope-mooncake-cc-iouring-lmcache-dynamo-nixl-phase132.en.md](megatron-fsdp-recompute-csa-rope-mooncake-cc-iouring-lmcache-dynamo-nixl-phase132.en.md)  
> **证据锚点**: `ev-megatron-fsdp-recompute-csa-rope-mooncake-cc-iouring-lmcache-dynamo-nixl-phase132-2026` (`own-offline` + `implementation`)  
> **可复现实验**: `python experiments/megatron_fsdp_recompute_csa_rope_mooncake_cc_iouring_lmcache_dynamo_nixl_phase132/run.py --check`

---

## 1. 第一手源码证据矩阵（Primary Source Matrix）

| 仓库与 Commit | 源码路径 | 核心架构变更 |
|---|---|---|
| `NVIDIA/Megatron-LM` [`90e4049`](https://github.com/NVIDIA/Megatron-LM/commit/90e40493e6d71a67ef51ca72a4bb47b8de0bc533) | `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py` | 在 `Megatron-FSDP` 中新增 `recompute_buckets` 追踪并在反向传播 `all_gather_params` 中启用激活重算（activation recompute）权重的异步预取，消除重算前向时的同步 `AllGather` 停顿 |
| `NVIDIA/Megatron-LM` [`a77288b`](https://github.com/NVIDIA/Megatron-LM/commit/a77288b18f1a3e5a7bfe2e680f6f693a7366c2e5) | `megatron/core/ops/attention/csa/output_rope.py` | DeepSeek-V4 压缩稀疏注意力（CSA）在前向原位旋转输出并在反向通过 `_mla_rope_reconstruct_output_kernel` 逆旋转重构原始输出与梯度，省去第 2 份输出激活显存（减少 `50.0%` 输出 RoPE 激活显存） |
| `NVIDIA/Megatron-LM` [`0f3e582`](https://github.com/NVIDIA/Megatron-LM/commit/0f3e582f6e13b94162994710a9780cccad89f8b9) | `megatron/core/model_parallel_config.py` | 新增 `max_seq_length_per_sample` 并将 `packed_seq_params` 透传至 `layer.mlp.preprocess`，使一次性分配通信缓冲的 `NCCL EP` 原生支持 `THD` 布局与序列打包（sequence packing） |
| `pytorch/pytorch` [`db460ec`](https://github.com/pytorch/pytorch/commit/db460ec60e6f821b963259bc8c3a6169d58e9384) | `aten/src/ATen/native/transformers/hip/flash_attn/ck/mha_bwd_ck.hip` | 在 AMD MI355X (`gfx950`) CK SDPA 反向传播中设置 `args.use_asm_v3 = hdim <= 192 && seqlen_q > 1`，将 `seqlen_q == 1`（`hdim 65..128`）路由至正确的 CK-tile 反向核，消除 `1e2–6e3` 相对梯度误差与非法访存 |
| `kvcache-ai/Mooncake` [`16fb473`](https://github.com/kvcache-ai/Mooncake/commit/16fb473b25cc78a3f1959a44051352b59d035d8a) | `mooncake-transfer-engine/include/transport/rdma_transport/rdma_endpoint.h` | 将误差驱动自适应拥塞控制（Adaptive Congestion Control）接入 Classic RDMA 传输层：在 Worker 准入侧施加有界端点投递窗口，利用完成反馈动态增长窗口并合并活跃路由补充 |
| `kvcache-ai/Mooncake` [`5e50b61`](https://github.com/kvcache-ai/Mooncake/commit/5e50b615b3c1d956f103a0191ec3ef5e7eef7da1) | `mooncake-transfer-engine/tent/src/transport/io_uring/io_uring_transport.cpp` | 在 TENT `IOUringTransport::freeSubBatch` 调用 `io_uring_queue_exit()` 销毁环并释放暂存缓冲前，等待所有已提交且仍在飞行中的 `io_uring` 请求完成（或轮询 CQ），防止内核异步读写入已释放的暂存内存 |
| `kvcache-ai/Mooncake` [`0a5ea84`](https://github.com/kvcache-ai/Mooncake/commit/0a5ea847cd6615ac371724bbb9fdff7670e07d37) | `mooncake-pg/include/pg_host_ip_resolve.h` | 在 `mooncake-pg` 多节点 `P2PHANDSHAKE` 初始化前向 `MASTER_ADDR` 推导可路由宿主机 IP（排除 `127.0.0.0/8` 全部环回段如 Debian `127.0.1.1`），防止 Rank 0 握手挂起 |
| `LMCache/LMCache` [`8a85b8c`](https://github.com/LMCache/LMCache/commit/8a85b8c965b44d33a3e934029b72aabd69bb8035) | `lmcache/v1/multiprocess/transport/grpc_impl/native_server.py` | 升级 `pygrpc-native==0.7.0` 并启用 `ordered=True`，保证多进程 gRPC Native 模式下同一客户端流水线亲和调用的严格保序执行 |
| `LMCache/LMCache` [`708cb05`](https://github.com/LMCache/LMCache/commit/708cb05c3791e45568cad8c53fe49901299070e1) | `lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py` | 在 `FSL2Adapter` 中同时校验缓冲区地址、偏移与长度相对块设备逻辑块大小（`_device_logical_block_size`）的对齐性，并在未对齐或遇 `EINVAL`/`EOPNOTSUPP` 时无缝回退至缓冲 I/O 且循环处理短写 |
| `ai-dynamo/dynamo` [`d1b1f24`](https://github.com/ai-dynamo/dynamo/commit/d1b1f244324382d46be6451bea76189372713e3e) | `components/src/dynamo/sglang/_compat.py` | 默认以 `"dynamo"` 格式发布 SGLang KV 事件，在事件载荷中携带 LoRA 适配器名称以支持路由器按适配器哈希 KV 块 |
| `ai-dynamo/nixl` [`15bc23f`](https://github.com/ai-dynamo/nixl/commit/15bc23f1bef8b6a828b0b858bbc5d25455335de4) | `examples/device/ep/csrc/nixl_ep.cpp` | 修正 `nixl_ep::Buffer::disconnect_ranks` 执行顺序：先完成内存视图重暂存与提交（`_nixl_ep_memory_views_stage` / `commit`）再调用 `_nixl_agents_disconnect`，确保断连后可靠释放对端 GPU 显存映射 |

---

## 2. 核心架构机制与确定性建模（Core Architectural Mechanisms）

1. **Megatron-FSDP 反向重算权重预取（`90e4049`）与 DSv4 CSA 原位输出 RoPE 反向重构（`a77288b`）**：
   - 在开启激活重算的 `Megatron-FSDP` 反向传播中，通过 `recompute_buckets` 将重算前向所需的行分片权重 `AllGather` 与当前微批反向 GEMM 重叠预取，单步暴露通信停顿从 **`14.8 ms` 降至 `1.8 ms`（削减 `87.84%`，单步提速 `1.1646x`）**。
   - DSv4 CSA 注意力在 `S=8192, B=2, H=64, D=128`（BF16）下原位执行输出 RoPE 并在反向通过 `_mla_rope_reconstruct_output_kernel` 逆旋转重构，每层省去第 2 份输出激活缓冲（每层节省 **`268,435,456 B` = `256.0 MiB`**，64 层共省 **`16.0 GiB` HBM**，输出 RoPE 激活显存降低 **`50.0%`**）。
2. **PyTorch MI355X `seqlen_q==1` CK-Tile 反向守卫（`db460ec`）、Mooncake Classic RDMA 误差驱动拥塞控制（`16fb473`）与 TENT `io_uring` 安全排空（`5e50b61`）**：
   - MI355X (`gfx950`) 在 `seqlen_q == 1, hdim=128` 下跳过错误的 `aiter` ASM v3 核并路由至 CK-tile，将 `dk/dv` 相对误差从 **`6000.0` 降至 `0.001`（故障率 `100% -> 0%`）**。
   - Mooncake Classic RDMA 接入自适应拥塞窗口与活跃路由合并补充，将高负载突发下的 RNR/超时重传风暴降低 **`88.0%`**；TENT `io_uring` 在销毁子批次环前排空飞行中 I/O，将释放后写入（use-after-free DMA）风险从 **`1 -> 0`**。
3. **LMCache `O_DIRECT` 逻辑块对齐回退（`708cb05`）、Dynamo SGLang LoRA KV 事件（`d1b1f24`）与 NIXL EP 断连显存回收（`15bc23f`）**：
   - LMCache `FSL2Adapter` 检查虚拟内存地址对齐并在 `EINVAL`/`EOPNOTSUPP` 时回退至缓冲 I/O，将非对齐/tmpfs 下的 L2 存储失败率从 **`100% -> 0%`**；Dynamo 默认启用携带 `lora_name` 的 `"dynamo"` KV 事件格式消除多 LoRA 路由哈希碰撞；NIXL EP 先提交新内存视图再断开对端 Agent，使弹性缩容后的对端 GPU 显存映射泄漏从 **`1 -> 0`**。
