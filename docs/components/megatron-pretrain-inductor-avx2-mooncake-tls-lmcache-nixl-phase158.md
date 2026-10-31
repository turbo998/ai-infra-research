# Megatron-LM PretrainConfig 数据容器与 Inductor 进程池防死锁、PyTorch AVX2 `zero_mask` 16 位掩码修复、Mooncake etcd TLS/mTLS 与无 GIL 内存注册、LMCache 真实落盘 Token 统计及 Dynamo/NIXL 遥测降维（Phase 158）

> **English Version**: [megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-phase158.en.md](megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-phase158.en.md)
> **关联问题 ID**：`q-megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-phase158` (`priority: 167`)
> **关联证据 ID**：`ev-megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-2026-phase158`

---

## 1. 架构背景与核心发现

在超大规模预训练与分布式 KV 缓存底座中，配置验证与运行时服务初始化的次序反转、CPython `fork()` 与 TorchInductor 异步编译池的关停竞态、AVX2 半精度零掩码位宽翻倍、RDMA 内存注册持有 Python GIL、以及 Prometheus 多进程指标带入 `pid` 标签，都会引发生产级故障。本报告基于 2026-10-12 至 2026-10-31 合并的 10 个主线提交，系统归纳以下五项工程机制：

1. **Megatron-LM `PretrainConfigContainer` 数据配置段接入、验证后初始化运行时服务与 Inductor 编译池防死锁（`Megatron-LM@831aa66`, `#8341`; `Megatron-LM@ce852bf`, `#8272`; `Megatron-LM@b607c1b`, `#8318`）**：将 `GPTDatasetConfig` 与 `DataloaderConfig` 纳入 `PretrainConfigContainer` 并在 tokenizer 就绪后回填验证；将 `initialize_runtime_services` 严格排在 `parse_and_validate_args` 之后；针对 Python `< 3.12.8` 及 `3.13.0` 的 `fork()` 竞态设置 `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0`，将 `async_compile.wait` 死锁率降为 **0.0%**。
2. **PyTorch AVX2 `Vectorized<Half/BFloat16>::zero_mask` 16 位压缩掩码修复与 ROCm `LoadHIP` 编译器保留（`pytorch@43ae9ac`, `#202480`; `pytorch@dc674ee`, `#201449`）**：此前 `_mm256_cmpeq_epi16` 后直接调用 `_mm256_movemask_epi8` 会为每个 16-bit 元素生成 2 个 bit（32-bit 掩码，7 个零元素误报 `popcount = 14`）；修复后先提取高低 128-bit 通道并通过 `_mm_packs_epi16(lo, hi)` 压缩为 16 字节再调用 `_mm_movemask_epi8`，严格输出 16-bit 掩码（1 bit/元素，`popcount = 7`）。同时修复 `LoadHIP.cmake` 覆盖用户预设 `CMAKE_HIP_COMPILER` 导致全量重编译的问题。
3. **Mooncake 环境变量驱动 etcd TLS/mTLS 与 `registerMemory` / `unregisterMemory` 释放 GIL（`Mooncake@99d2614`, `#4945`; `Mooncake@c72982f`, `#4019`）**：新增 `MOONCAKE_ETCD_TLS_*` 环境变量覆盖 Store HA、快照、保活与 Transfer Engine 的全部 6 个 etcd 客户端入口（保留原 hostname 作 gRPC `ServerName` 并拒绝 `https://` 静默降级，`13/13` CTest 用例通过）；在 `TransferEnginePy::registerMemory` 与 `unregisterMemory` 中加入 `pybind11::gil_scoped_release`，将单次 RDMA 内存注册期间（`~18.50 ms`）的 Python RPC/心跳线程阻塞抖动降至 **0.12 ms**（降低 **154.17x**）。
4. **LMCache `MP_STORE_END` 仅统计真实新增落盘 Chunk 的 Token 数与 Dynamo `qwen3_coder` 解析器升级（`LMCache@12dfa32`, `#6188`; `dynamo@5a48166`, `#17676`）**：修复 `storage_manager.commit_put_tasks` 返回已缓存 chunk 导致日志虚高的问题，改为仅统计至少成功写入一个新对象的唯一键位置 `stored_chunks`（在 16 个 chunk 中有 12 个已命中缓存、4 个新写入的场景下，剔除 **75.0%** 的虚报 token 数，`4096 -> 1024 tokens`）；同时将 `dynamo-parsers-v2` 升级至 `0.8.9` 修复 `qwen3_coder` `<tool_call>` 触发前缀自排斥缺陷。
5. **NIXL `prometheus_mp` 仅在同名冲突时附加 `pid` 与 `agent_instance` 标签（`nixl@250581c`, `#2387`）**：仅当同一 `hostname + agent_name + local_rank` 下存在 $\ge 2$ 个存活进程时才输出 `pid` 与 `agent_instance` 标签，在 64 个 Agent 经历 5 次重启（累计 320 个 PID）的场景下将 Prometheus 时间序列基数从 `320` 压降至 `64`（缩减 **5.0x**）。

---

## 2. 权威上游证据矩阵

| 仓库与提交 | 关键模块路径 | 核心机制与定量指标 |
| :--- | :--- | :--- |
| `NVIDIA/Megatron-LM@831aa66` (`#8341`) | `megatron/training/config/config_container.py` | 将 `GPTDatasetConfig` 与 `DataloaderConfig` 纳入 `PretrainConfigContainer` 并支持延迟字段重校验 |
| `NVIDIA/Megatron-LM@ce852bf` (`#8272`) | `megatron/training/initialize.py` | 严格在 `parse_and_validate_args` 完成后再调用 `initialize_runtime_services` |
| `NVIDIA/Megatron-LM@b607c1b` (`#8318`) | `tests/unit_tests/conftest.py` | 在 Python `<3.12.8` 及 `3.13.0` 下设置 `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0`，死锁率 **0.0%** |
| `pytorch/pytorch@43ae9ac` (`#202480`) | `aten/src/ATen/cpu/vec/vec256/vec256_half_avx2.h` | `_mm_packs_epi16` + `_mm_movemask_epi8` 修复 `Half`/`BFloat16` 16-bit `zero_mask`（1 bit/元素） |
| `pytorch/pytorch@dc674ee` (`#201449`) | `cmake/public/LoadHIP.cmake` | 保留已设置的 `CMAKE_HIP_COMPILER`，避免 CMake 缓存清空与全量重编译 |
| `kvcache-ai/Mooncake@99d2614` (`#4945`) | `mooncake-common/etcd/src/etcd_helper.cpp` | 支持 `MOONCAKE_ETCD_TLS_*` TLS/mTLS 与 gRPC `ServerName` 校验（`13` 项 CTest 通过） |
| `kvcache-ai/Mooncake@c72982f` (`#4019`) | `mooncake-integration/transfer_engine/transfer_engine_py.cpp` | `registerMemory`/`unregisterMemory` 增加 `gil_scoped_release`，线程阻塞从 `18.50 ms` 降至 `0.12 ms`（**154.17x**） |
| `LMCache/LMCache@12dfa32` (`#6188`) | `lmcache/v1/multiprocess/server.py`, `storage_manager.py` | `MP_STORE_END` 仅按实际新增落盘的唯一 `stored_chunks` 计算 `num_tokens`（消除 **75.0%** 虚报） |
| `ai-dynamo/dynamo@5a48166` (`#17676`) | `Cargo.lock`, `lib/llm/Cargo.toml` | 升级 `dynamo-parsers-v2` 至 `0.8.9`，修复 `qwen3_coder` `<tool_call>` 触发器自排斥 |
| `ai-dynamo/nixl@250581c` (`#2387`) | `src/plugins/prometheus_mp/prometheus_mp_Engine.cpp` | 仅在 `hostname + agent_name + local_rank` 冲突时打 `pid` 标签，重启 5 次下序列基数降低 **5.0x** |

---

## 3. 可复现实验验证

```powershell
python experiments\megatron_pretrain_inductor_avx2_mooncake_tls_lmcache_nixl_phase158\run.py --check
```
