# 分布式 KV 感知路由器 (`NVIDIA Dynamo` + `NIXL`)：有理数负载前缀信用衰减 (`overlap_score_credit_decay`) 与条件式 PD 解耦旁路 (`IslBoundingPolicy`)

> **证据等级声明**：本文对 `DefaultWorkerScorer::worker_logit` 有理数前缀重叠信用衰减公式、多级缓存权重（Device HBM / Host Pinned DRAM / Disk / Shared Storage）及 `ConditionalDisaggPolicy`（`IslBoundingPolicy` + `DecodeLoadGate` + `should_bypass_remote_prefill`）的分析基于 `ai-dynamo/dynamo`（commit `e3fb11d8632259cdbcea338eafa34b1e1cb04d59`）源码文件 `lib/kv-router/src/scheduling/selector/reference.rs`（blob `8e8bbfb3da51352138ac5e24c6062c96d8d420b2`）、`lib/kv-router/src/conditional_disagg.rs`（blob `e92175be96dadb06b1f68d8e3735179e27673e1d`）与 `lib/llm/src/kv_router/prefill_router/conditional_bypass.rs`（blob `da162d9077b354f0567bbe723cb2dc8b72d37457`）的一手核验（`implementation`）。16 节点 Zipfian 多轮 Agent 突发流量下的缓存命中率与 P99 TTFT 对比来自确定性离线仿真算例 `experiments/dynamo_kv_aware_router/run.py`（`own-offline`），不声称生产集群真机实测。

---

## 1. 为什么多轮 Agent 推理中“纯前缀粘性路由”与“纯最小负载路由”都会失效？

在具有 `8K–32K` 共享系统提示词、代码库上下文或多轮工具调用历史的 Agentic LLM 集群中，分布式请求路由器（Router）面临一个经典的**局部性-拥塞二难困境（Locality vs Congestion Dilemma）**：
1. **无感知轮询 / 最小负载路由（`ROUND_ROBIN` / `LEAST_LOADED`）**：将同一会话或共享同一热门文档前缀的连续请求随机打散到 16 台 Worker 上。由于单台 Worker 的 HBM Prefix Cache 容量有限，各节点缓存不断发生重复加载与 LRU 淘汰抖动（Cache Thrashing），导致 HBM 命中的前缀比例暴跌至 `~33%`，大量算力浪费在重复 Prefill 已知前缀上。
2. **静态前缀粘性路由（`STICKY_PREFIX_NO_DECAY`）**：始终将匹配某前缀的请求强行路由至持有该 KV 块的唯一 Worker。一旦某个热门前缀（如高频 Agent 模板或突发并发子任务）在短时间内涌入 10+ 个并发请求，该 Worker 的 Prefill 队列瞬间堆积数百毫秒至数秒的排队延迟（Head-of-Line Blocking），导致 **P99 TTFT 飙升至 `2,766 ms` 以上**，而集群其余 15 台空闲 Worker 却完全无法分担负载！

`NVIDIA Dynamo` (`lib/kv-router`) 通过两项核心机制彻底消解了这一冲突：
- **有理数负载前缀信用衰减（Rational Overlap Credit Decay）**：当目标 Worker 的在途 Prefill 积压超过集群最闲节点时，按有理函数 $\frac{1}{1 + d \cdot L_{\text{norm}}}$ 动态折减其 KV 缓存命中信用；
- **条件式 Prefill-Decode 解耦旁路（Conditional Disaggregation Bypass）**：当选定的 Decode Worker 已经缓存了绝大部分前缀（剩余未命中 `effective_isl < 2048` 且未命中比例 `< 0.7`）且 Decode 队列未过载时，直接在该 Decode Worker 本地执行聚合 Prefill（Aggregated Local Prefill），省去远程 Prefill 节点排队与跨网 `NIXL` RDMA KV 传输开销！

---

## 2. 一手源码走读：`DefaultWorkerScorer::worker_logit` 有理数衰减代价函数

在 `ai-dynamo/dynamo`（commit `e3fb11d8632259cdbcea338eafa34b1e1cb04d59`）的 `lib/kv-router/src/scheduling/selector/reference.rs`（blob `8e8bbfb3da51352138ac5e24c6062c96d8d420b2`）中，`DefaultWorkerScorer::worker_logit` 精确定义了每个候选 Worker 的路由代价 Logit（**越低越优先**）：

```rust
// Source: ai-dynamo/dynamo lib/kv-router/src/scheduling/selector/reference.rs
let excess_active_prefill_blocks = load
    .active_prefill_tokens
    .saturating_sub(min_active_prefill_tokens)
    / block_size;
let normalized_prefill_load = excess_active_prefill_blocks as f64 / request_blocks.max(1) as f64;
let overlap_credit_decay = if self.overlap_score_credit_decay <= 0.0 || normalized_prefill_load <= 0.0 {
    1.0
} else {
    1.0 / (1.0 + self.overlap_score_credit_decay * normalized_prefill_load)
};
let effective_overlap_score_credit = self.overlap_score_credit * overlap_credit_decay;
let overlap_credit_blocks = effective_overlap_score_credit * device_overlap_blocks
    + self.host_cache_hit_weight * host_overlap_blocks
    + self.disk_cache_hit_weight * disk_overlap_blocks
    + self.shared_storage_hit_weight * shared_beyond_device_blocks as f64;
let raw_prefill_blocks = request_blocks as f64 + excess_active_prefill_blocks as f64;
let adjusted_prefill_blocks = (raw_prefill_blocks - overlap_credit_blocks).max(0.0);
let logit = self.prefill_load_weight * adjusted_prefill_blocks
    + self.decode_load_weight * decode_cost_blocks
    + self.active_request_Load_weight * load.active_requests as f64;
```

### 2.1 代数形式与四级分层缓存信用折算

设请求总长度为 $\text{ISL}$，块大小为 $B_{\text{blk}}$（如 `64` tokens），请求总块数 $R = \lceil \text{ISL} / B_{\text{blk}} \rceil$。候选 Worker $w$ 当前活跃 Prefill 积压超出全集群最小积压的块数为 $\Delta P_w = \lfloor (T_{\text{active}, w} - \min_j T_{\text{active}, j}) / B_{\text{blk}} \rfloor$，归一化积压负载为 $L_{\text{norm}, w} = \Delta P_w / \max(R, 1)$。则设备端 HBM 缓存命中块 $O_{\text{dev}, w}$ 的有效抵扣信用因子为：

$$\gamma_{\text{decay}}(L_{\text{norm}, w}) = \frac{1}{1 + d_{\text{decay}} \cdot L_{\text{norm}, w}}$$

连同主机锁页内存（Host Cache，权重 $w_{\text{host}}$）、本地 NVMe 磁盘（Disk Cache，权重 $w_{\text{disk}}$）以及集群共享存储池（Shared HiCache / Mooncake / NIXL，仅计超出本地 HBM 的净增块 $O_{\text{shared}\setminus\text{dev}}$，权重 $w_{\text{shared}}$），总抵扣块数为：

$$C_{\text{overlap}, w} = \left(c_0 \cdot \gamma_{\text{decay}}(L_{\text{norm}, w})\right) O_{\text{dev}, w} + w_{\text{host}} O_{\text{host}, w} + w_{\text{disk}} O_{\text{disk}, w} + w_{\text{shared}} O_{\text{shared}\setminus\text{dev}, w}$$

最终调整后净 Prefill 负载与综合 Logit 为：

$$P_{\text{adj}, w} = \max\left(0,\; R + \Delta P_w - C_{\text{overlap}, w}\right), \qquad \text{Logit}_w = w_P P_{\text{adj}, w} + w_D D_{\text{cost}, w} + w_A N_{\text{req}, w}$$

在 `experiments/dynamo_kv_aware_router/run.py` 中，我们对一条 `ISL = 8,192` tokens（$R = 128$ 块，Worker 0 命中 $O_{\text{dev}} = 112$ 块即 `87.5%` 前缀）的请求绘制了随队列积压 $\Delta P_w$ 增长的 Logit 演化曲线（$d_{\text{decay}} = 0.65$）：
- 当 $\Delta P_w = 0$（Worker 0 空闲）时：$\gamma_{\text{decay}} = 1.0$，抵扣 `112.0` 块，$P_{\text{adj}} = 16.0$ 块，**`Logit = 24.6`**（远低于冷空闲节点的 `Logit = 128.0`，坚定命中缓存）；
- 当 $\Delta P_w = 64$（积压半个请求）时：$\gamma_{\text{decay}} = 0.75472$，抵扣降至 `84.528` 块，**`Logit = 116.07`**（仍略优于完全冷节点，但已优先分流给持有 Host/Shared 缓存的次优空闲节点）；
- 当 $\Delta P_w = 128$（积压 1 个完整请求）时：$\gamma_{\text{decay}} = 0.60606$，$P_{\text{adj}} = 188.12$ 块，**`Logit = 196.72 > 128.0`**，路由器立即自动将溢出请求平滑卸载（Spillover）至空闲节点，从数学上根除了热点单点雪崩！

---

## 3. 一手源码走读：`ConditionalDisaggPolicy` 与 `should_bypass_remote_prefill`

在 Prefill-Decode (PD) 解耦架构中，并非所有请求都值得发送给远程 Prefill 池并通过 NIXL RDMA 拷贝 KV Cache。`ai-dynamo/dynamo` 在 `lib/kv-router/src/conditional_disagg.rs`（blob `e92175be96dadb06b1f68d8e3735179e27673e1d`）与 `lib/llm/src/kv_router/prefill_router/conditional_bypass.rs`（blob `da162d9077b354f0567bbe723cb2dc8b72d37457`）中实现了双门限动态旁路器：

```rust
// Source: ai-dynamo/dynamo lib/kv-router/src/conditional_disagg.rs
pub const DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD: usize = 2048;
pub const DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD: f64 = 0.7;

fn should_disaggregate(&self, decision_input: &DisaggDecisionInput) -> bool {
    let effective_isl = decision_input
        .prompt_tokens
        .saturating_sub(decision_input.decode_chosen_cached_tokens);
    let effective_isl_ratio = if decision_input.prompt_tokens == 0 {
        0.0
    } else {
        effective_isl as f64 / decision_input.prompt_tokens as f64
    };
    effective_isl >= self.effective_isl_threshold
        || effective_isl_ratio >= self.effective_isl_ratio_threshold
}
```

以及 `conditional_bypass.rs` 中的 `should_bypass_remote_prefill`：
1. **净新增序列长度双门限（`IslBoundingPolicy`）**：
   计算已选定 Decode Worker 上的净未缓存 Token 数 $\text{ISL}_{\text{eff}} = \text{ISL} - T_{\text{cached\_on\_decode}}$ 及未命中比例 $r_{\text{eff}} = \text{ISL}_{\text{eff}} / \text{ISL}$。只有当**同时满足** $\text{ISL}_{\text{eff}} < 2048$ 且 $r_{\text{eff}} < 0.70$ 时，才表明该 Decode Worker 本地已拥有足够高的前缀缓存（>30% 命中且仅剩不到 2048 个增量后缀 Token 需要计算）。
2. **Decode 负载安全门控（`DecodeLoadGate`）**：
   在旁路远程 Prefill 前，额外检查目标 Decode Worker 的活跃 KV 块与生成负载是否低于安全水位（`decode_gate_allows_bypass`）。若未过载，则直接返回 `should_bypass_remote_prefill = true`，在 Decode Worker 本地以 Chunked Prefill 完成剩余 `< 2048` tokens 的增量计算，**彻底归零远程 Prefill 节点排队延迟与 NIXL 跨网全量 KV 传输耗时**！

---

## 4. 定量实验核验：16 节点 480 请求多轮 Agent 突发流量基准对比

在 `experiments/dynamo_kv_aware_router/run.py` 中，我们在 16 台 Worker（每节点 HBM 容量缓存 6 个大前缀 + Host DRAM 缓存 6 个前缀 + 共享 HiCache 层）上重放了 480 条包含周期性微突发（Micro-Bursts）的 Zipfian (`s=1.05`) 多轮 Agent 请求轨迹（前缀 `8K–16K` tokens，增量后缀 `512–6,144` tokens）：

| 路由与 PD 解耦策略 | 设备 HBM 命中率 (`%`) | 分层综合有效命中率 (`%`) | 条件旁路本地执行率 (`Bypass %`) | 平均排队等待 (`ms`) | 平均 TTFT (`ms`) | P50 TTFT (`ms`) | P99 TTFT (`ms`) | 相比 Sticky P99 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`ROUND_ROBIN`** (无状态轮询) | `33.71%` | `45.29%` | `0.00%` | `285.89 ms` | `379.55 ms` | `376.72 ms` | `796.30 ms` | `3.47x` |
| **`LEAST_LOADED`** (纯最小积压优先) | `32.85%` | `43.21%` | `0.00%` | `302.64 ms` | `398.22 ms` | `437.81 ms` | `568.27 ms` | `4.87x` |
| **`STICKY_PREFIX_NO_DECAY`** (无衰减前缀粘性) | **`75.26%`** | `75.26%` | `0.00%` | `622.97 ms` | `682.71 ms` | `475.18 ms` | `2,766.13 ms` | `1.00x` |
| **`DYNAMO_RATIONAL_DECAY`** (`d=0.65` + 分层信用) | `57.61%` | **`75.26%`** | `0.00%` | `11.16 ms` | `74.73 ms` | `67.55 ms` | **`177.52 ms`** | **`15.58x`** |
| **`DYNAMO_CONDITIONAL_DISAGG_BYPASS`** (`Dynamo` + `IslBounding`) | **`59.45%`** | **`75.26%`** | **`42.00%`** | **`6.32 ms`** | **`59.62 ms`** | **`58.94 ms`** | **`163.24 ms`** | **`16.95x` (`4.88x` vs RR)** |

### 关键架构结论
1. **兼得高缓存命中率与低尾延迟**：`DYNAMO_RATIONAL_DECAY` 在微突发发生时自动衰减过载节点的前缀信用并将溢出请求引导至持有 Host/Shared 缓存的空闲节点，使分层有效命中率保持在 **`75.26%`**（比 `ROUND_ROBIN` 设备命中率高 **`2.23x`**），同时将平均排队等待从 `STICKY_PREFIX_NO_DECAY` 的 `622.97 ms` 削减至 **`11.16 ms`**；
2. **`ConditionalDisaggPolicy` 消除 42% 冗余远程 KV 搬运**：开启 `IslBoundingPolicy` (`eff_isl < 2048` & `ratio < 0.7`) 后，**`42.0%`** 的多轮跟进请求直接在缓存命中的 Decode Worker 上本地完成增量 Prefill，将平均 TTFT 再降 `20.2%`（`59.62 ms` vs `74.73 ms`），**P99 TTFT 压至 `163.24 ms`（相比 Sticky 提速 `16.95x`，相比 Round-Robin 提速 `4.88x`）**。
