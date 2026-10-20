# Mooncake 页聚合批量 RDMA 传输 API、LMCache Scratch 组零跨度跳过与 NVIDIA Dynamo MultiConnector 二级 KVCR 发现

> **English Version**: [mooncake-page-gather-batch-transfer-lmcache-scratch-skip-dynamo-kvcr-tiers.en.md](mooncake-page-gather-batch-transfer-lmcache-scratch-skip-dynamo-kvcr-tiers.en.md)

## 1. 一手源码证据与核心定位

| 系统 / 仓库 | 精确 Commit / 日期 | 核心源码路径 | 证据等级 |
|---|---|---|---|
| **Mooncake** (`kvcache-ai/Mooncake`) | `387aecb27cc9268be8318c421972ee5e9125a12e` (`2026-10-19`, PR `#4763`) | `mooncake-integration/transfer_engine/transfer_engine_py.cpp`, `transfer_engine_py.h` | `implementation` |
| **LMCache** (`LMCache/LMCache`) | `2a34fa829b3c902c5b6393740239f6ff65bd0ff8` (`2026-10-19`, PR `#5818`) | `lmcache/integration/vllm/lazy_offload_manager.py`, `tests/v1/test_lazy_offload_manager.py` | `implementation` |
| **NVIDIA Dynamo** (`ai-dynamo/dynamo`) | `662cb22df401d8a5285383f83e7defca34cab227` (`2026-10-20`, PR `#16778`) | `components/src/dynamo/vllm/kv_hints.py`, `components/src/dynamo/vllm/tests/test_kv_hints.py` | `implementation` |

---

## 2. Mooncake TransferEngine：Buffer-Protocol 与页聚合 (`_pages`) 批量传输 API (`#4763`)

### 2.1 长序列 PD 分离 KV 传输中的 pybind11 逐元素拆箱瓶颈
在 Prefill-Decode (PD) 分离推理中，单条长提示词的 KV Cache 跨节点传输包含数万个离散描述符（$F$ 个层/字段字段 $\times$ $P$ 个分页 Block）。旧版 `batch_transfer_sync_write` 接收 3 个 Python `list[int]`（`buffers`、`peer_buffer_addresses`、`lengths`），导致调用方在 Python 层构造 $3 \times F \times P$ 个整数对象，再由 pybind11 逐个拆箱转换为 `std::vector`。

### 2.2 `_arrays` 零循环拷贝与 `_pages` C++ 笛卡尔展开
`387aecb2` 在 `TransferEnginePy` 中新增两类零拆箱接口：
1. **`batch_transfer_{sync,async}_{write,read}_arrays`**：通过 `pybind11::buffer` 直接校验 1D C-contiguous 8 字节整数数组（格式码 `QLqlNn`，对有符号 `qln` 校验非负）并执行单次 `std::memcpy`。
2. **`batch_transfer_sync_{write,read}_pages`**：接收长度为 $P$ 的 `src_pages`、`dst_pages` 与形状为 $[F, 5]$ 的字段表 `fields`（每行依次为 `src_base, src_stride, dst_base, dst_stride, length`），在 C++ 内按 **Field-Major（字段优先）** 顺序展开全部 $F \times P$ 个传输描述符：
   $$\text{src\_addr}(f, i) = \text{src\_base}_f + \text{src\_pages}[i] \times \text{src\_stride}_f, \quad \text{dst\_addr}(f, i) = \text{dst\_base}_f + \text{dst\_pages}[i] \times \text{dst\_stride}_f$$
   并按 `max_batch_size` 分块调用 `batchTransferSync`。
- **量化收益**：当 $F=64, P=512$（共 $32,768$ 个描述符）时，Python 侧传入的 8B 整数数量从 $3 \times 32,768 = 98,304$ 降至 $2 \times 512 + 5 \times 64 = 1,344$（**跨语言参数体积削减 $98.6328\%$**），并以 `max_batch_size=8192` 拆分为 4 个同步批次提交。

---

## 3. LMCache：`LazyOffloadManager` 混合 KV 组中的 Scratch 组跳过 (`#5818`)

### 3.1 混合模型 Scratch Ring Group (`tokens_per_block = 0`) 的除零异常
在 Qwen3.8-Flash-Next、GLM-5.3-Flash 等混合架构模型中，vLLM 會在可缓存 KV 组之间配置一个 **Scratch Ring Group**（每个请求生命周期内仅固定持有 1 个 Block，不随调度 Token 增长，因此向连接器报告 `tokens_per_block = 0`，例如 `_group_tokens_per_block = [16, 0, 32]`）。
修复前，`LazyOffloadManager._drain()` 估算下一拍 Block 压力时直接计算：
```python
est_next_step_blocks = sum(
    -(-scheduler_output.total_num_scheduled_tokens // tokens_per_block)
    for tokens_per_block in self._group_tokens_per_block
)
```
遇到 `tokens_per_block == 0` 立即抛出 `ZeroDivisionError`。

### 3.2 正跨度过滤 (`if tokens_per_block > 0`)
`2a34fa82` 增加守卫条件 `if tokens_per_block > 0`：对于 `group_tokens_per_block = [16, 0, 32]`、调度 Token 数 $T_{\text{sched}} = 48$，精确估算下一拍新增可缓存块数为 $\lceil 48/16 \rceil + \lceil 48/32 \rceil = 3 + 2 = 5$。

---

## 4. NVIDIA Dynamo：穿透 `MultiConnector` 发现二级 `kvcr` 控制端点 (`#16778`)

在 vLLM 同时挂载 `NixlConnector`（PD 传输）与 `OffloadingConnector`（分层卸载）时，顶层连接器被包装为 `MultiConnector`（位于 `MULTI_CONNECTOR_WRAPPERS` 中）。旧版 `dynamo/vllm/kv_hints.py` 的 `_secondary_tiers()` 仅读取顶层 `extra_config.get("secondary_tiers")`，导致嵌套在 `extra_config["connectors"][i]["kv_connector_extra_config"]["secondary_tiers"]` 中的 `kvcr` 二级存储层被漏检。
`662cb22d` 展开 `MULTI_CONNECTOR_WRAPPERS` 子连接器配置列表，正确提取并向 Dynamo Router 发布 `kv_hint_transfer = true` 与控制面端点 `{"0": "tcp://127.0.0.1:17771"}`。

---

## 5. 可复现离线验证

```bash
python experiments/mooncake_page_gather_lmcache_scratch_dynamo_kvcr/run.py --check
```
