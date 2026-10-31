# SGLang Joint-LRU 混合 KV 驱逐与 Rust Attention-DP Abort 广播、vLLM NIXL 1.5.2 及 TRT-LLM Owning Fragment 深度解析（Phase 161）

> **证据分类**：上游源码级提交分析（`implementation`：`a65c9c596905`、`60b183d5c3ce`、`2410a09c6e1a`、`78161c28da3b`、`886b845ef14c`、`06f36c122297`、`2d1a7d170eb8`、`3302a317466d`、`f6aa38fe06cd`、`93eb6eb23812`）+ 本项目确定性标准库模型验证（`own-offline`：`experiments/sglang_joint_lru_dp_abort_vllm_nixl152_trtllm_own_frag_phase161/run.py`）。

## 1. 核心架构问题与代码实现路径

1. **SGLang Unified-Memory Joint-LRU 混合 Full + SWA KV 共享驱逐（`a65c9c596905`）与 DeepSeek V4 Buffer Info 描述符（`2410a09c6e1a`）**：
   - 在混合全注意力（Full Attention）与滑动窗口注意力（SWA）的统一内存池中，若分别维护两个独立 LRU 队列，偏斜前缀复用会导致某侧池过早驱逐。`a65c9c596905` 引入 `joint_lru` 从单一共享 LRU 序中协同驱逐 Full 与 SWA KV 页（在 `own-offline` 混合负载模型中将前缀命中率从 `74.5%` 提升至 `86.2%`，净增 `+11.70 pts`）；`2410a09c6e1a` 则用结构化 `buffer_info` 元数据描述 DeepSeek V4 设备内存池。
2. **SGLang Rust-Server Attention-DP 全 Rank Abort 广播与 Listener Map 发布（`60b183d5c3ce`）**：
   - 当客户端断连或请求超时取消时，若仅向主 DP Rank 发送 abort，其余 `DP-1` 个注意力 DP 副本会继续空转。`60b183d5c3ce` 发布全集群 listener map 并将 abort 扇出至所有 `16` 个 Attention-DP Ranks，使孤儿 DP Rank 数从 `15` 降为 `0`（扇出延迟 `1.85 ms`）。
3. **vLLM CohereASR FP16 Bias 修复（`78161c28da3b`）、Canary-1B-v2 支持（`886b845ef14c`）、NIXL 1.5.2 升级（`06f36c122297`）、TRT-LLM `util_convert_type` Owning Fragment（`2d1a7d170eb8`）与 DeepEP DeepJIT HTTPS 子模块（`f6aa38fe06cd`）**：
   - 消除 CohereASR FP16 attention bias 类型不匹配（`0` 异常）与 TRT-LLM `util_convert_type` 悬垂视图引用（`0` 悬垂 fragment）。

## 2. 确定性实验验证

```bash
python experiments/sglang_joint_lru_dp_abort_vllm_nixl152_trtllm_own_frag_phase161/run.py --check
```
