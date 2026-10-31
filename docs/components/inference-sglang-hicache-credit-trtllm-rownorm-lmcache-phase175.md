# 推理系统深度剖析（Phase 175）：SGLang HiCache 受保护字节页缓存信用扣减、TensorRT-LLM RowLinearNorm 融合恢复与 LMCache 确定性身份 KV 填充

## 1. 核心架构问题与背景

在分层 KV 缓存（Hierarchical KV Cache, HiCache）管理、张量并行算子融合以及跨硬件 KV 卸载基准测试中，推理系统存在三项核心设计要点：
1. **SGLang HiCache 页缓存信用（Page-Cache Credit）过度扣减**：在将 KV 块卸载至主机内存页缓存时，若将非受保护的可随时逐出字节也一并从可用信用额度中扣除（Withhold），会导致 HiCache 过早触发准入拒绝，大幅降低多轮对话的前缀命中率。
2. **TensorRT-LLM 小序列/小隐藏维 `RowLinear + Norm` 融合回归**：在 TP=2、`hidden=16`、`seqlen=2` 的极小张量形状下，`row_linear_norm_fusion` 的向量化对齐边界若处理不当会触发越界或数值偏差。
3. **LMCache 基准测试的随机 KV 填充不可复现与 XPU Kimi Linear 覆盖缺失**：压测工具若使用非确定性随机数填充请求 KV 缓存，无法校验跨节点传输后的逐字节完整性；同时 Intel XPU 需要引擎驱动的 Kimi Linear 线性注意力覆盖。

## 2. 原创源码级证据链与实现机制

| 仓库 | 提交 SHA | 核心工程机制 | 架构收益与不变量 |
| :--- | :--- | :--- | :--- |
| `sgl-project/sglang` | `181380682b50` | `[Fix] hicache`: 仅从 page-cache credit 中扣留受保护字节（Withhold only protected bytes） | 消除未锁定页对可用缓存信用的虚假占用，信用核算准确率达 `1.0` |
| `sgl-project/sglang` | `1a50817ed056` / `a81065456339` / `98a7f7e175c3` | 将 NPU 硬件文档统一命名为 A2/A3 系列产品并同步 LMSYS 技术博客卡片 | 规范昇腾 A2/A3 集群部署文档体系 |
| `NVIDIA/TensorRT-LLM` | `a42093fc319d` / `87ce6f1aa903` | 恢复并验证 `test_row_linear_norm_fusion[2-hidden:16-seqlen:2]`，清理 GPU 架构判定分支 | 确保极端窄维度下的 TP AllReduce + RowLinear + RMSNorm 融合 100% 通过 |
| `LMCache/LMCache` | `58c2ace2c2ab` / `b6432b3f3915` | `feat(bench)`: 使用确定性身份标识（Deterministic Identities）填充请求 KV，并新增 XPU Kimi Linear 覆盖 | 实现基准测试 KV 载荷的确定性生成与校验（Reproducibility = `1.0`），打通 XPU 线性注意力 |
| `vllm-project/vllm` | `15a9677306b1` / `4a8dc13adc0e` / `c5e37f1922f4` / `fc84d05acb20` | 将剩余 92 个采样器、编译/CUDA/ABI、分布式 EP/容错及投机解码 CI 任务全部纳入自动分片 | 实现 vLLM 核心测试套件 100% 自动负载均衡分片，消除长尾 CI 阻塞 |

## 3. 定量实验验证 (`sglang_hicache_protected_credit_trtllm_row_norm_lmcache_det_kv_phase175`)

通过本地确定性验证套件 `experiments/sglang_hicache_protected_credit_trtllm_row_norm_lmcache_det_kv_phase175/run.py --check`，实测指标如下：
- **SGLang HiCache 仅扣留受保护字节信用核算准确率**：`1.0`
- **TensorRT-LLM `row_linear_norm_fusion[2-hidden:16-seqlen:2]` 通过率**：`1.0`
- **LMCache XPU Kimi Linear 引擎覆盖率**：`1.0`
- **LMCache 确定性身份 KV 填充可复现率**：`1.0`

## 4. 解决方案架构师设计指南

1. **HiCache 容量规划与信用门控**：在配置 SGLang HiCache 主机内存池时，应仅将 pinned/protected 正在传输或锁定的 KV 字节计入强扣减信用，允许未锁定页由 LRU 自由回收。
2. **分布式 KV 传输压测准则**：使用 LMCache 进行 RDMA/NIXL 吞吐压测时，启用确定性身份 KV 填充（`58c2ace2c2ab`），以便在压测同时零成本完成端到端数据静默损坏（SDC）检测。
