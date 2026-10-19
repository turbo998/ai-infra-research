# UEC 数据包裁剪 (`Packet Trimming`) 与浅缓冲交换机动态 $\alpha$ 准入控制：800G/1.6T Incast 破局架构

> **证据等级声明**：本文对 SONiC `BUFFER_POOL`、`BUFFER_PROFILE`（`dynamic_th`、`xoff`、`xon_offset`）、`WRED_PROFILE` 与 `PFC_WD` 的分析基于 `sonic-net/sonic-buildimage`（commit `3e670e64f9c4b48a5af3964d2e6dff0295c48dda`）源码文件 `src/sonic-yang-models/tests/files/sample_config_db.json`（blob `54a3708beb31359361f92ec826f682fd0f65fefd`）与 `src/sonic-yang-models/yang-models/sonic-buffer-profile.yang`（blob `6f7154f99b7d66b609e751aa4d9ba4db0724d862`）的一手核验（`implementation`），结合 Ultra Ethernet Consortium (UEC) 1.0 规范（`specification`）。800G 浅缓冲交换机在 8..64 并发 Incast 下的队列吸收水位、PFC 暂停端口数与尾延迟对比来自确定性仿真算例 `experiments/uec_packet_trimming_buffer/run.py`（`own-offline`）。

---

## 1. 为什么在 800G/1.6T 浅缓冲 AI 交换机上 RoCEv2 无损 `PFC` 走向物理极限？

在高密度 800 Gbps / 1.6 Tbps AI 交换芯片（如 Broadcom Tomahawk 5 `51.2 Tbps` / Tomahawk 6 `102.4 Tbps`）中，片上 SRAM MMU 缓冲区的增长速度远慢于端口SerDes带宽的翻倍速度：
- 以 SONiC 标准配置 `src/sonic-yang-models/tests/files/sample_config_db.json`（commit `3e670e64f9c4`）为例，单个入向无损缓冲池 `ingress_lossless_pool` 的物理容量为 **`33,004,032 Bytes`（`31.48 MiB`）**：

```json
// Source: sonic-net/sonic-buildimage src/sonic-yang-models/tests/files/sample_config_db.json
"BUFFER_POOL": {
    "ingress_lossless_pool": {
        "mode": "static",
        "size": "33004032",
        "xoff": "196608",
        "type": "ingress"
    }
},
"BUFFER_PROFILE": {
    "pg_lossless_100000_300m_profile": {
        "dynamic_th": "1",
        "pool": "ingress_lossless_pool",
        "size": "9427",
        "xoff": "50176",
        "xon": "0",
        "xon_offset": "3584"
    }
}
```

### 1.1 `XOFF` 无损吸收预留（Headroom）的线速膨胀税

在 RoCEv2 无损以太网中，当上游交换机或网卡收到 PFC `XOFF` 暂停帧前，链路上已经在途的比特（In-flight bits during cable RTT + MAC/PHY/FEC pipeline）必须由入向端口的专用 `xoff` Headroom 无损吞下：

$$B_{\text{xoff\_per\_port}}(R, L) = R \cdot \left( \frac{2 L}{v_{\text{fiber}}} + T_{\text{MAC+FEC+Pause}} \right)$$

- 在 `100 Gbps`、`300m` 光纤下，SONiC `pg_lossless_100000_300m_profile` 需预留 `xoff = 50,176 Bytes`。
- 当端口速率升至 **`800 Gbps`（`100 KB/us`）** 时，单端口 `xoff` 预留暴增约 8 倍至 **`393,216 Bytes`（`384 KiB`）**！在单 ITM 管辖的 `32` 个 800G 端口上，仅为了保证“不丢包”而锁死的静态/半静态 XOFF Headroom 就高达 **`12.58 MB`（占整个 `33.00 MB` 入向池的 `38.13%`）**，导致真正可用于吸收突发拥塞的共享动态池缩水至 `20.42 MB`。

### 1.2 SONiC `dynamic_th` ($\alpha = 2^k$) 动态阈值公式

在 Broadcom SAI / SONiC 中，`dynamic_th = k`（取值为 `-8..3`）定义了单队列最大可用深度 $T_q(t)$ 与当前共享池剩余空闲空间 $B_{\text{free}}(t)$ 的比例系数 $\alpha = 2^k$。当有 $N_{\text{active}}$ 个拥塞队列同时竞争容量为 $B_{\text{usable}}$ 的共享池时，稳态单队列阈值与池吸收率满足：

$$T_q = \frac{\alpha}{1 + N_{\text{active}} \cdot \alpha} B_{\text{usable}}, \qquad \text{Utilization}_{\text{pool}} = \frac{N_{\text{active}} \cdot \alpha}{1 + N_{\text{active}} \cdot \alpha}$$

在 RoCEv2 下，由于 `38.13%` 的池容量被 `XOFF` Headroom 冻结，即使将 `dynamic_th` 设为 `1`（$\alpha = 2.0$，4 活跃队列），实际能调动的缓冲也仅占总 MMU 池的 **`55.00%`（`17,726.67 KiB`）**。

---

## 2. UEC `Packet Trimming`（数据包裁剪）与 0.5-RTT 极速反馈机制

Ultra Ethernet Consortium (UEC 1.0) 彻底摒弃了对正常业务流的 `PFC` 无损依赖，转而通过 **数据包裁剪（Packet Trimming）+ 高优控制队列（CP Queue）+ 多路径源路由喷洒（EVB）** 实现免无损（Lossless-Free）低尾延迟传输：

```
[MoE All-to-All / Checkpoint 32-to-1 Incast Burst @ 800 Gbps]
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Switch Egress Queue Depth Q(t) vs Dynamic Threshold T_q(t) = α·B_free  │
├───────────────────────────────────┬─────────────────────────────────────┤
│ RoCEv2 (PFC + DCQCN, α=2):        │ UEC 1.0 (Packet Trimming, α=1):     │
│ 1. ECN marked packet waits in     │ 1. Zero XOFF reserve (100% pool     │
│    deep queue -> 1.0 RTT (12 us)  │    usable = 33.0 MB).               │
│    before sender gets CNP.        │ 2. When Q(t) > T_trim (1.5 MB),     │
│ 2. Ingress pool hits XOFF limit ->│    switch TRIMS 4096B payload to    │
│    broadcasts PFC PAUSE upstream. │    a 64B header-only CP packet!     │
│ 3. 29/32 ports paused; innocent   │ 3. 64B CP header bypasses data      │
│    victim flows suffer 24.7x HoL! │    queue -> arrives in 0.5 RTT (6us)│
│ 4. Risk of PFC_WD (100 ms drop).  │ 4. Precise window cut + selective   │
│                                   │    retransmit; 0 PFC paused ports!  │
└───────────────────────────────────┴─────────────────────────────────────┘
```

1. **零 XOFF 预留释放 `+61.6%` 动态共享池容量**：无需 PFC 后，`33,004,032 Bytes` 共享池 100% 可用于动态拥塞吸收；在 `dynamic_th = 0`（$\alpha = 1.0$）下，4 个活跃队列即可调动 **`25,784.40 KiB`（占总池 `80.0%`）** 的有效缓冲。
2. **从 `4096 B` 载荷丢弃到 `64 B` 报头裁剪（Header Trimming）**：当出向队列达到浅裁剪阈值（`1.5 MB`）时，交换机流水线硬件直接切掉 `4096 B` 数据载荷，仅保留 **`64 B` 包头** 并打上 `Trimmed` 标记，立即转入**最高优先级的控制平面队列（CP Queue）**转发给接收端 NIC。
3. **0.5-RTT（`6.0 us`）极速闭环 vs RoCEv2 1.0-RTT（`12.0 us` + 队列排队延迟）**：在 RoCEv2 中，被打上 ECN 标记的大包必须先排完数兆字节的数据队列才能到达接收端并生成 CNP；而在 UEC 中，`64 B` 裁剪包头走高优队列绕过数据排队，在 **`0.5 RTT`（`6.0 us`）** 内触发发送端精准扣减拥塞窗口并仅对被裁剪的 PSN 发起选择性重传（Selective Retransmission）。

---

## 3. 定量仿真对比：`8..64` 并发 Incast 下的集合通信 CCT 与受害流尾延迟

基于 `experiments/uec_packet_trimming_buffer/run.py` 的确定性仿真结果（单发送端突发 `512 KiB`，端口速率 `800 Gbps`）：

| Incast 并发发送端数 ($N_{\text{senders}}$) | 突发总数据量 (`MiB`) | 理想串行化耗时 (`us`) | RoCEv2 PFC 暂停端口数 | RoCEv2 P99 CCT (`us`) | UEC 裁剪包数 (`64B CP`) | UEC P99 CCT (`us`) | **UEC CCT 加速比** | RoCEv2 受害流 P99 (`us`) | UEC 受害流 P99 (`us`) | **受害流尾延迟消减倍数** |
|---|---|---|---|---|---|---|---|---|---|---|
| **8** | `4.0` | `41.9` | `3` | `181.8` | `0` | `59.9` | **3.03x** | `57.6` | `15.5` | **3.73x** |
| **16** | `8.0` | `83.9` | `12` | `450.7` | `56` | `104.2` | **4.32x** | `194.4` | `16.4` | **11.84x** |
| **32** | `16.0` | `167.8` | `29` | `971.9` | `515` | `207.2` | **4.69x** | `452.8` | `18.3` | **24.69x** |
| **64** | `32.0` | `335.5` | `63` | `2014.2` | `1434` | `413.2` | **4.87x** | `969.6` | `22.2` | **43.72x** |

### 核心架构师设计准则
- **MoE All-to-All（`32-to-1` / `64-to-1` Incast）**：RoCEv2 因 ECN 反馈滞后与 ECMP 哈希极化触发大面积 PFC 暂停（29–63 个端口被锁死），导致无辜背景流（如心跳、参数服务器控制流或相邻任务流）遭受 **24.69x–43.72x** 的队头阻塞（HoL Blocking）尾延迟爆炸；而 UEC `Packet Trimming + EVB` 将集合通信完成时间（CCT）缩短 **4.69x–4.87x**，且受害流 P99 延迟始终稳定在 **`15.5–22.2 us`**。
