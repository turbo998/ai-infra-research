# SGLang 逻辑页交错 KV 分片（`KVShard`）与 Prefill/Decode 分离传输契约：`filter_kv_indices_for_shard_rank` 模商重映射、`max_bootstrap_input_len` 准入边界与全分片拉取拓扑

> **English Version**: [sglang-kv-shard-pd-transfer-page-interleave.en.md](sglang-kv-shard-pd-transfer-page-interleave.en.md)  
> **证据等级 (Evidence Class)**: `implementation`（一手上游源码核验）+ `own-offline`（`experiments/sglang_kv_shard_pd_transfer/run.py` 标准库确定性复算）

---

## 1. 架构背景与核心问题

在 DeepSeek-V3/R1 等采用 **MLA（Multi-head Latent Attention）** 的大模型推理中，标准 Attention TP 会让每个 TP Rank 冗余保存完整的一份潜变量 KV 缓存（$c_t^{KV} \in \mathbb{R}^{512}$ + $k_t^{R} \in \mathbb{R}^{64}$），单卡 KV 容量成为长上下文并发的首要瓶颈。`sgl-project/sglang` 通过逻辑页交错 KV 分片（`--enable-kv-cache-sharding`，`KVShard`）将逻辑 KV 页 $l$ 按模 $N$ 分散存储在各 Prefill TP Rank 上（Rank $r = l \bmod N$ 仅保存本地物理页 $l_{\text{local}} = \lfloor l / N \rfloor$），从而将总逻辑 KV 容量扩大 $N$ 倍。然而，当 `KVShard` 与 **Prefill/Decode (P/D) 分离架构** 结合时，暴露出三大关键工程挑战：

1. **分块传输页归属与全局偏移重映射**：在 Chunked Prefill 或UnifiedRadixCache 旋转页链下，待发送的逻辑页数组 `kv_indices` 并不满足位置同余，若发送端仍使用连续切片 `index_slice` 或未跳过本 Rank 零归属的非末尾分块，将导致 Decode 端 KV 错位崩溃。
2. **MLA P/D 单点拉取失效与 DCP 冲突**：传统 MLA P/D 下每个 Decode TP Rank 仅向同序号的单个 Prefill TP Rank 拉取 KV；但开启 `KVShard` 后没有任何单个 Prefill Rank 拥有完整 KV，必须切换为全分片多对多拉取，且与 Decode Context Parallelism (`DCP`) 的序列切分语义互斥。
3. **Bootstrap 队列尾部容量死锁（300 秒 Decode 超时）**：`KVShard` 下 `add_one_req_ignore_eos`（关闭 Radix Cache 的最严路径）需要为页向上取整、每个分片预留 1 页以及 2 个额外 Token 留出余量。若 Bootstrap 队列直接按 $N \times \text{capacity\_per\_rank}$ 放行请求，落在尾部缺口内的长 Prompt 将永远无法通过分片调度器准入，最终触发 Decode 端 **300 秒 KV 等待超时**。

---

## 2. 一手源码实现核验 (`implementation`)

### 2.1 `filter_kv_indices_for_shard_rank` 模商重映射与空分块跳过 (`eb4b167d27`, `#38473`)

在 `python/sglang/srt/disaggregation/utils.py`（blob `44a4f111c24b226349fc6b6510a8f919dcf8e792`）与 `python/sglang/srt/disaggregation/common/conn.py`（blob `7b6012eebe29f3f7707c32361ecc5e310a5f6463`）中：

```python
def filter_kv_indices_for_shard_rank(
    *,
    kv_indices: np.ndarray,
    index_slice: slice,
    shard_rank: int,
    shard_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Logical page l lives on rank l % N as local page l // N; returns the owned
    local ids and their send positions, which replace the contiguous index_slice."""
    owned = kv_indices % shard_size == shard_rank
    positions = np.arange(index_slice.start, index_slice.stop, dtype=np.int64)
    return kv_indices[owned] // shard_size, positions[owned]
```

- 在 `CommonKVSender._prepare_send_indices` 中，当 `self.kv_mgr.kv_shard_size > 1` 时，调用 `filter_kv_indices_for_shard_rank` 将连续 `index_slice` 替换为离散的 `positions[owned]`，并在 `len(kv_indices) == 0 and not is_last_chunk` 时直接返回 `(kv_indices, index_slice, is_last_chunk, True)` 跳过当前非末尾分块的 RDMA 传输，仅在 `is_last_chunk == True` 时保留完成通知信号。

### 2.2 MLA 全分片拉取映射与 DCP 互斥护栏 (`conn.py` blob `7b6012eebe`)

- 在 `CommonKVManager.try_ensure_parallel_info` 中，若 Decode 开启了 `DCP`（`self.dcp_size > 1`）且 Prefill 携带 `info.enable_kv_cache_sharding == True`，立即抛出 `RuntimeError("PD decode DCP does not support a prefill with --enable-kv-cache-sharding.")`。
- 在 `CommonKVManager._resolve_rank_mapping` 中，当 `self.is_mla_backend and info.enable_kv_cache_sharding and info.attn_cp_size == 1` 时，强制要求 `self.attn_tp_size == info.attn_tp_size`，并将：
  - `target_tp_rank = None`
  - `target_tp_ranks = list(range(info.attn_tp_size))`
  - `required_dst_info_num = self.attn_tp_size`
  - `required_prefill_response_num = info.attn_tp_size`
  使得每个 Decode Rank 从全部 $N$ 个 Prefill TP Rank 收集各自的 $1/N$ 页分片并拼回完整 KV。

### 2.3 `max_bootstrap_input_len` 精确准入边界与双针长文验证 (`prefill.py` blob `8c5fd2c374` & `test_disaggregation_kv_cache_sharding.py` blob `11fc1cf565`)

在 `python/sglang/srt/disaggregation/prefill.py` 中：

$$\text{max\_bootstrap\_input\_len}(C, P, N) = \begin{cases} C, & N = 1 \\ \left\lfloor \dfrac{C - N \cdot P - 3}{P} \right\rfloor \cdot P, & N > 1 \end{cases}$$

其中 $C = N \cdot C_{\text{rank}}$ 为扩展后的逻辑总 Token 容量，$P$ 为 `page_size`，$N$ 为 `shard_size`。`PrefillBootstrapQueue._check_if_req_exceed_kv_capacity` 使用 `self.max_input_len` 替代 `self.max_total_num_tokens`，在请求进入 Bootstrap 队列瞬间直接返回 `HTTP 400 BAD_REQUEST` 拒绝无法调度的尾部超长 Prompt。

在 `test/registered/e2e/disaggregation/test_disaggregation_kv_cache_sharding.py`（blob `11fc1cf565`）与 `page_interleave_pool.py`（`e7f7447333` blob `f4aab3973e`）/ `kv_shard_hook.py`（`de487f8039` blob `8203c4652d`）中：
- 单 Rank 物理池容量 `PREFILL_POOL_TOKENS = 16,384`（`page_size = 64`，即 `256` 物理页/Rank），`prefill_tp_size = 2` 时总逻辑容量为 `32,768` Token（`512` 逻辑页），Bootstrap 上限为 `32,576` Token；
- 构造 `LONG_PROMPT_TOKENS = 24,576`（单 Rank 物理池的 **1.50 倍**，总 KV 字节大于单 Rank 池的 `1.3x`），将 `"red"` 密钥置于逻辑页 `20`（由 Rank `20 % 2 = 0` 保存在本地页 `10`）、`"blue"` 密钥置于逻辑页 `341`（Token 偏移 `21,824 > 16,384`，由 Rank `341 % 2 = 1` 保存在本地页 `170`），验证 Decode 从两个 Prefill Rank 拉取后 100% 恢复双针密钥。

---

## 3. 标准库确定性离线实验 (`own-offline`)

通过 `python experiments\sglang_kv_shard_pd_transfer\run.py --check` 对旋转页链重映射、分块空片跳过、Bootstrap 边界与双针长文恢复进行复算：

| 分片数 $N$ (`shard_size`) | 单 Rank 物理容量 ($C_{\text{rank}}$) | 聚合逻辑容量 ($C = N \cdot C_{\text{rank}}$) | `max_bootstrap_input_len` 准入上限 | 拦截的不可调度尾部缺口 ($C - \text{limit}$) | 避免的 Decode 等待超时 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `1` (Unsharded) | `16,384` | `16,384` | **`16,384`** | `0` tokens | `0 s` |
| `2` (`TP=2` KVShard) | `16,384` | `32,768` | **`32,576`** | **`192` tokens** (`[32577..32768]`) | **`300 s`** |
| `4` (`TP=4` KVShard) | `16,384` | `65,536` | **`65,216`** | **`320` tokens** (`[65217..65536]`) | **`300 s`** |
| `8` (`TP=8` KVShard) | `16,384` | `131,072` | **`130,496`** | **`576` tokens** (`[130497..131072]`) | **`300 s`** |

- **旋转页链无损重构**：对 9 页旋转序列 `[2, 7, 8, 13, 18, 23, 24, 29, 34]`（`index_slice = slice(20, 29)`，$N=4$），4 个分片分别过滤出本地页 `[2, 6]`、`[3, 7]`、`[0, 4, 8]`、`[1, 5]` 与全局发送位置 `[22, 26]`、`[23, 27]`、`[20, 24, 28]`、`[21, 25]`，Decode 端按 $(l_{\text{local}} \cdot N + r)$ 重构实现 **100% 精确还原**。
- **双针跨分片恢复**：`24,576` Token Prompt（单卡池 **1.50x**）中，`red` 页 `20` 落于 Rank `0` 本地页 `10`，`blue` 页 `341` 落于 Rank `1` 本地页 `170`，验证了无任何单卡保存全量副本下的 P/D 零丢失传输。

---

## 4. 架构师设计准则与可证伪问题

1. **何时启用 MLA `KVShard`**：在 Prefill 显存受限且 `attn_cp_size == 1`、`dcp_size == 1`、Prefill 与 Decode `attn_tp_size` 相等的 Blackwell (`trtllm_mla`) 集群上启用 `--enable-kv-cache-sharding`，可在不增加 HBM 占用的前提下将 Prefill 逻辑 KV 容量提升 $N$ 倍。
2. **网关限流对齐**：前端 API 网关的最大输入 Token 校验必须对齐 `max_bootstrap_input_len(capacity, page_size, shard_size)`（扣除 $N \cdot P + 3$ 的向下页取整），切勿直接配置为 $N \cdot C_{\text{rank}}$。
3. **后续可证伪问题**：当 Decode 端同样引入逻辑页分片或非对称 TP（$TP_{\text{prefill}} \neq TP_{\text{decode}}$）时，`filter_kv_indices_for_shard_rank` 如何扩展为 $(N_{\text{src}}, N_{\text{dst}})$ 二维重映射矩阵而不增加 ZMQ 握手往返？
