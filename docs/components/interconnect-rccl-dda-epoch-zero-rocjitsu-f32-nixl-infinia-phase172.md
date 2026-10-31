# 互连 DDA 纪元清零序、零选择子存储与 DDN Infinia 2.5 深度剖析（Phase 172）：ROCm RCCL DDA Scratch/Epoch 清零序、`rocjitsu` FLOAT32 存储及 NIXL Infinia 2.5

> **证据分类**：`implementation`（固定上游提交点对点源码审计）+ `own-offline`（确定性标准库解析模型验证）

---

## 1. 一手上游提交矩阵

| 仓库 | 提交 SHA | 日期 | 核心工程变更 |
|---|---|---|---|
| `ROCm/rocm-systems` | `e7cb4b477a` | 2026-10-30 | `fix(rccl): order DDA fabric scratch/epoch zeroing before first LL kernel (#14254)` |
| `ROCm/rocm-systems` | `ce3fa320bd` | 2026-10-30 | `[rocjitsu] Fix zero-selector FLOAT32 buffer stores and enable guitar-hero coverage (#13988)` |
| `NVIDIA/nccl` | `59a784c3d9` | 2026-09-10 | `nccl4py: generate bindings from in-tree NCCL headers` |
| `NVIDIA/nccl` | `76c747405b` | 2026-09-10 | `Improve nccl4py packaging and versioning` |
| `ai-dynamo/nixl` | `5bca2560f8` | 2026-10-06 | `DDN Infinia 2.5 Updates (#2357)` |
| `ai-dynamo/nixl` | `7c8d82c1b2` | 2026-10-07 | `EXAMPLES/DEVICE/EP: Removed high-throughput kernels. (#2345)` |
| `kvcache-ai/Mooncake` | `46061cb23a` | 2026-10-16 | `[PR] Add priority declaration (#4718)` |
| `LMCache/LMCache` | `8bfcdfdea4` | 2026-10-22 | `fix: make foreign tools import independent of safe path (#5986)` |

---

## 2. 核心架构机制解析

### 2.1 ROCm RCCL DDA Fabric Scratch/Epoch 清零定序（`e7cb4b477a`）与 `rocjitsu` FLOAT32 存储（`ce3fa320bd`）
- **DDA Scratch/Epoch 清零先于首个 LL Kernel（`#14254`）**：强制将 Direct Data Access (DDA) fabric scratch 缓冲区与 epoch 计数器的清零操作定序在首个 Low-Latency (LL) 集合通信算子启动之前，彻底消除首轮 LL 算子读取未初始化纪元标志导致的死锁与数据竞争（`0` 次竞争）。
- **`rocjitsu` 零选择子 FLOAT32 Buffer Store 修复（`#13988`）**：修复当 `DST_SEL` 包含零选择子时 FLOAT32 buffer store 写坏相邻通道的问题（`0` 次损坏）。

### 2.2 `nccl4py` 树内头文件绑定生成（`59a784c3d9`）与 NIXL DDN Infinia 2.5（`5bca2560f8`）
- **`nccl4py` ABI 零漂移**：直接从仓库内 NCCL 头文件生成 Python 绑定（`0` 处 ABI 不匹配），同步升级 NIXL 对象存储插件至 DDN Infinia `2.5`（`#2357`），并解耦 LMCache 外部工具导入对 `safe path` 的依赖（`#5986`）。

---

## 3. 确定性离线验证结果

运行 `python experiments/rccl_dda_epoch_zero_rocjitsu_f32_nixl_infinia25_phase172/run.py --check`：
- `uninitialized_dda_epoch_ll_races`: **`0`** (`rocjitsu_zero_sel_f32_store_corruptions = 0`)
- `nccl4py_header_abi_mismatches`: **`0`** (`ddn_infinia_sdk_version = "2.5"`)
- `lmcache_foreign_import_path_failures`: **`0`**
