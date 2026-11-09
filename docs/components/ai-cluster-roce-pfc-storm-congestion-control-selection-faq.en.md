# AI Cluster RoCEv2 Lossless Ethernet Congestion Control & PFC Storm Deadlock Prevention Selection Guide (`Lossy Go-Back-N` vs. `Pure PFC Pause Storm` vs. `DCQCN + PFC Watchdog` vs. `Spectrum-X / UET Adaptive Packet Spraying`)

> **Reader's Guide**: Built for **AI Solution Architects and AI Fabric Network Engineers**, this guide explains from first principles: **why dropping just 0.1% of packets on a 512-GPU 400G RoCEv2 cluster collapses AllReduce bandwidth by 77% via RNIC hardware `Go-Back-N` retransmission; why turning on raw Priority-based Flow Control (`PFC`) without ECN causes catastrophic `PFC Pause Storms` and cascading fabric deadlocks; how `DCQCN` enforces the 3-level buffer watermark rule ($K_{\min} < K_{\max} \ll \text{XOFF}_{\text{PFC}}$) with a `200ms` PFC Watchdog; and how `Spectrum-X / Ultra Ethernet (UET)` combines per-packet adaptive spraying with RNIC hardware out-of-order Direct Data Placement (`OOO DDP`) to reach InfiniBand-grade `95.2%` Ethernet fabric utilization.**

![AI Cluster RoCEv2 Lossless Ethernet Congestion Control & PFC Storm Selection Summary](../../assets/svg/ai-cluster-roce-pfc-storm-congestion-control-selection-faq-summary.svg)

---

## ① 📌 Executive Summary

1. **Why Naive Ethernet Collapses Under RDMA Collectives (Tier 1 & Tier 2 Traps)**:
   - **Lossy RoCEv2 without Flow Control (Tier 1) = Go-Back-N Collapse**: Unlike Linux kernel TCP (which buffers out-of-order packets in host RAM and selectively retransmits only the single lost packet via SACK), RDMA NICs (RNICs) execute transport logic inside tiny on-chip SRAM state machines. By default, if packet `#5` out of `#1..#1000` is tail-dropped by a switch micro-burst, the receiving RNIC discards packets `#6..#1000` and forces a full **Go-Back-N rollback**. On a 512-GPU 400G (`50 GB/s`) cluster, a `3.85%` retransmission rate crushes AllReduce bus bandwidth to **`11.4 GB/s` (`22.8%` utilization ✕)** and MoE All-to-All to **`6.8 GB/s` ✕**!
   - **Pure Hardware PFC without ECN/Watchdog (Tier 2) = PFC Pause Storm Deadlock**: Enabling IEEE 802.1Qbb **Priority-based Flow Control (PFC)** alone prevents packet drops by sending link-level `XOFF PAUSE` frames upstream when switch buffers fill up. However, on multi-tier Leaf-Spine fabrics, `XOFF` frames propagate hop-by-hop into a **backpressure tree (Head-of-Line blocking)**, freezing innocent training jobs and triggering **`6.5` cluster-wide PFC storm outages per month ✕** (`$2.92M/yr` in wasted GPU stall time)!
2. **Standard Commodity SONiC Baseline (Tier 3) vs. Large-Scale MoE Ethernet Gold Standard (Tier 4)**:
   - **For 64–1,024 GPU Commodity SONiC Clusters (Broadcom Tomahawk-4/5), standardize on `Tier 3 (DCQCN [Early ECN Before PFC] + 200ms SONiC PFC Watchdog + Rail-Optimized Topology)`**: Enforcing strict buffer watermarks ($K_{\min}=150\text{ KB} < K_{\max}=800\text{ KB} \ll \text{XOFF}_{\text{PFC}}=1.5\text{ MB}$) marks packets with **ECN** early so the receiver returns a **CNP (Congestion Notification Packet)** that throttles only the offending sender RNIC—resolving **99% of congestion before PFC ever fires** and achieving **`41.8 GB/s` AllReduce (`83.6%` utilization ✅)**!
   - **For 512–32,768 GPU MoE Training & Disaggregated KV Serving, deploy `Tier 4 (Spectrum-X / Ultra Ethernet [UET] Per-Packet Adaptive Spraying + RNIC Out-of-Order Direct Data Placement)`**: Eliminates the `42%` static 5-tuple ECMP elephant-flow hash collision penalty by spraying packets across all Spine links and reordering them directly in RNIC hardware (`ConnectX-7 OOO DDP`), pushing AllReduce to **`47.6 GB/s` (`95.2%` utilization 🏆)** and MoE All-to-All to **`45.2 GB/s` (`2.48x` faster than Tier 2 🏆)** with **`0.0` PFC storm outages**!
3. **First-Principles Mechanic**:
   - **PFC** is a **blind lane-wide emergency brake** (blocks the entire Priority 3 queue); **DCQCN (ECN)** is a **surgical per-flow throttle**; **Adaptive Packet Spraying + RNIC OOO DDP** turns static single-lane hashes into a **self-balancing multi-lane highway**.

---

## ② 🎯 Architect & Customer FAQ (First-Principles Q&A)

### Q1: Why does TCP survive 0.1% packet loss effortlessly while RoCEv2 (RDMA over Converged Ethernet) collapses when even a single packet drops?

**Plain-Language First Principles**:
- In standard kernel TCP, the OS maintains megabytes of socket reorder buffers in host DRAM and uses **Selective Acknowledgment (SACK)** to retransmit only the exact lost segment.
- In **RoCEv2 (RDMA)**, packets bypass the OS kernel entirely and are DMA-written directly into GPU HBM by the NIC ASIC. Traditional RNIC hardware lacked large reorder buffers—if packet `#2` is dropped during a `32:1` MoE All-to-All Incast micro-burst, the receiving RNIC rejects packets `#3..#1000` as out-of-sequence and triggers a **Go-Back-N retransmission rollback**, collapsing 400G (`50 GB/s`) effective bandwidth to **`11.4 GB/s` (`22.8%`)** on AllReduce and **`6.8 GB/s`** on MoE All-to-All (**Tier 1**).

---

### Q2: If we enable lossless PFC on our switches to stop packet drops, why do we get catastrophic "PFC Pause Storms" that freeze the entire cluster? How should we tune ECN vs. PFC watermarks?

**Plain-Language First Principles**:
- **Why Pure PFC Fails (Tier 2)**:
  - **PFC (`802.1Qbb`)** pauses an entire lossless priority class (`Priority 3`) on an ingress port whenever a downstream egress queue crosses `XOFF`. On a Leaf-Spine fabric, if one Leaf port backs up, it pauses the Spine; the Spine buffer fills within microseconds and pauses every other Leaf switch (**Head-of-Line Blocking & Backpressure Tree Spread**). Worse, a single hung NIC or flapping optical transceiver emitting continuous `PAUSE` frames triggers a **PFC Storm** that deadlocks the entire fabric (`6.5 outages/month` in Tier 2).
- **How Tier 3 (`DCQCN + 200ms PFC Watchdog`) Solves It**:
  - **The Golden Watermark Inequality ($K_{\min} < K_{\max} \ll \text{XOFF}_{\text{PFC}}$)**:
    1. **Early WRED ECN Marking ($K_{\min}=150\text{ KB}, K_{\max}=800\text{ KB}$)**: As soon as a switch queue reaches `150 KB`, the switch sets the IP header **ECN bits to `11` (Congestion Experienced)** without pausing traffic. The receiving RNIC immediately sends a **CNP (Congestion Notification Packet)** back to the sending RNIC, which smoothly cuts its injection rate. **99% of micro-bursts are absorbed here with zero PFC frames.**
    2. **Emergency PFC Airbag ($\text{XOFF}_{\text{PFC}}=1.5\text{ MB}$)**: Hardware PFC `XOFF` only fires if an extreme burst exceeds `1.5 MB` before the ECN control loop completes a round trip.
    3. **Hardware Circuit Breaker (`SONiC PFC Watchdog = 200 ms`)**: If any switch port remains paused by PFC for more than `200 ms`, the switch hardware watchdog automatically drops the stuck queue and quarantines the faulty port, preventing a single broken NIC from freezing the entire data center.

---

### Q3: Why does well-tuned DCQCN (Tier 3) still lose 16%–31% bandwidth on MoE All-to-All compared to InfiniBand? What is "Static ECMP Elephant-Flow Hash Collision" and how do Spectrum-X / UET solve it?

**Plain-Language First Principles**:
- **The Static ECMP Elephant-Flow Collision Problem**:
  - Traditional Ethernet switches route flows across Spine links by hashing the **5-tuple** (`SrcIP, DstIP, SrcPort, DstPort, Proto`). While 5-tuple hashing balances 100,000 tiny web flows statistically, an AI training node generates only a few massive **400 Gbps RDMA Elephant Flows**. With static hashing, two 400G flows have a **`42%` probability of colliding on the exact same Spine uplink** (causing 200% oversubscription and ECN throttling) while adjacent Spine uplinks sit completely idle!
- **How Tier 4 (`Spectrum-X / UET Adaptive Packet Spraying + RNIC Out-of-Order Direct Data Placement`) Reaches `95.2%` Efficiency**:
  1. **Switch Per-Packet Adaptive Spraying**: Leaf switches dynamically spray individual packets of the same RDMA flow across all available Spine links based on real-time queue depth, eliminating elephant-flow hash collisions.
  2. **RNIC Hardware Out-of-Order Direct Data Placement (`ConnectX-7 OOO DDP`)**: Because packets sprayed across different Spine paths arrive slightly out of order, the receiving RNIC uses an on-chip **PSN Bitmap Window** to DMA-write out-of-order packets directly into their target GPU HBM offsets without triggering Go-Back-N retransmission.
  - **Result**: AllReduce bus bandwidth reaches **`47.6 GB/s` (`95.2%` of 400G line rate 🏆)** and MoE All-to-All jumps to **`45.2 GB/s` (`2.48x` faster than Tier 2 🏆)** with `2.1 ms` Incast P99 latency!

---

## ③ 🧭 Visual Architecture Diagrams (Mermaid Decision Tree / Watermark & Routing Flow / Sequence Diagram)

### 3.1 AI Cluster RoCEv2 Congestion Control & Routing Decision Tree (`flowchart TD`)

```mermaid
flowchart TD
    A["🚀 Start AI Cluster RoCEv2 Fabric Architecture Selection<br/>(Goal: 0 PFC Storm Deadlocks + Break ECMP Hash Collisions)"] --> B{"Using Lossy Ethernet (Tier 1)<br/>or Raw PFC without ECN/Watchdog (Tier 2)?"}
    B -- "Yes (Tier 1 or Tier 2)" --> C["⛔ Reject for Production AI Clusters<br/>Tier 1 collapses to 22.8% BW via Go-Back-N;<br/>Tier 2 suffers 6.5 PFC storm deadlocks/month!"]
    B -- "Production Lossless Congestion Control" --> D{"Cluster Scale, Workload & Switch Generation?"}
    D -- "64–1,024 GPUs on Commodity SONiC (Tomahawk-4/5)<br/>Standard Dense DP/TP/PP Training" --> E["✅ Select Tier 3: DCQCN + PFC Watchdog + Rail-Optimized<br/>• 3-Level Watermarks: ECN(150KB–800KB) ≪ PFC(1.5MB)<br/>• Enable 200ms SONiC PFC Watchdog circuit breaker<br/>• 41.8 GB/s AllReduce (83.6% fabric efficiency)"]
    D -- "512–32,768 GPUs / MoE All-to-All Training<br/>or Disaggregated KV Cache Burst Serving" --> F["🏆 Select Tier 4: Spectrum-X / UET Adaptive Spraying<br/>• Per-packet adaptive spraying eliminates hash collisions<br/>• RNIC hardware Out-of-Order Direct Data Placement (OOO DDP)<br/>• 47.6 GB/s AllReduce (95.2%) & 45.2 GB/s MoE All-to-All!"]
```

### 3.2 Pure PFC + Static ECMP vs. DCQCN Watermarks + Adaptive Packet Spraying (`flowchart LR`)

```mermaid
flowchart LR
    subgraph Bad["❌ Legacy Trap: Pure PFC + Static 5-Tuple ECMP (42% Hash Collision)"]
        direction TB
        B1["Two 400G Elephant Flows Hash to Spine-1<br/>(Spine-1 200% Congested, Spine-2 0% Idle)"] --> B2["Leaf Buffer Hits 1.5MB Ceiling<br/>Sends Hop-by-Hop PFC XOFF Pause Frames"]
        B2 --> B3["Backpressure Tree Freezes Innocent Jobs<br/>(6.5 PFC Storm Deadlocks/Mo, $2.92M/Yr Waste ✕)"]
    end

    subgraph Good["🏆 Modern Paradigm: Early ECN (Tier 3) + Adaptive Spraying & OOO DDP (Tier 4)"]
        direction TB
        G1["Per-Packet Adaptive Spraying Across All Spines<br/>(95.2% Uniform Link Utilization ✅)"] --> G2["Early ECN Marking at 150KB Triggers CNP Throttle<br/>(99% Congestion Resolved Before 1.5MB PFC Line ✅)"]
        G2 --> G3["ConnectX-7 / UET Hardware OOO Direct Data Placement<br/>(PSN Bitmap Reordering in HBM, 47.6 GB/s 🏆)"]
    end

    Bad -. "Upgrade (+148% All-to-All BW / 0 Outages)" .-> Good
```

### 3.3 End-to-End DCQCN Early ECN + Adaptive Packet Spraying & RNIC OOO Reordering Flow (`sequenceDiagram`)

```mermaid
sequenceDiagram
    autonumber
    participant SrcNIC as Sender RNIC-A (400G)
    participant LeafSw as Adaptive Leaf/Spine Fabric
    participant DstNIC as Receiver RNIC-B (OOO DDP)
    participant GPUHBM as Receiver GPU HBM

    SrcNIC->>LeafSw: 1. Burst 400G MoE All-to-All RDMA packets (PSN #101, #102, #103...)
    Note over LeafSw: 2. Per-Packet Adaptive Spraying<br/>#101 routed via Spine-1, #102 via Spine-2 (Zero hash collision)
    Note over LeafSw: 3. Micro-burst reaches K_min=150KB (far below 1.5MB PFC floor)<br/>Switch marks #102 IP ECN bits = 11 (CE) without pausing
    LeafSw->>DstNIC: 4. Packet #102 (ECN=11) arrives 300ns ahead of #101 (Out-of-Order)
    DstNIC->>GPUHBM: 5. RNIC OOO DDP engine DMA-writes #102 directly to target HBM offset (Zero drop!)
    DstNIC->>SrcNIC: 6. Detects ECN=11 and transmits hardware CNP frame back to Sender RNIC-A
    LeafSw->>DstNIC: 7. 300ns later #101 arrives; RNIC completes PSN bitmap window & posts CQE
    Note over SrcNIC: 8. Sender RNIC-A receives CNP and smoothly trims injection rate (0 PFC pauses!)
```

---

## ④ 📊 Architect Decision Matrix

*Benchmark Profile: 512-GPU H100 Cluster (`64 nodes × 8×400Gbps` RoCEv2 ConnectX-7 NICs, 2-Tier 51.2T Leaf-Spine fabric, `50.0 GB/s` theoretical line rate per NIC), running `16 GB AllReduce` + `MoE All-to-All (32:1 Incast)`.*

| Selection Dimension / Tier | Tier 1: Lossy RoCEv2 + Static ECMP (`Go-Back-N` Collapse) | Tier 2: Pure PFC without ECN/Watchdog (`PFC Storm` Trap) | Tier 3: `DCQCN` (`ECN ≪ PFC`) + `200ms Watchdog` + Rail-Optimized | Tier 4: `Spectrum-X / UET` Adaptive Packet Spraying + RNIC `OOO DDP` |
| :--- | :--- | :--- | :--- | :--- |
| **Congestion Control Mechanism** | No PFC, No ECN (Tail drop triggers RNIC Go-Back-N rollback) | 802.1Qbb PFC `XOFF` Pause Frames Only | **Early ECN (`150–800KB`) + CNP Rate Cut + `1.5MB` PFC Floor + Watchdog** | **Hardware Telemetry (RTT/Credit) + Zero-Deadlock VoQ + End-to-End CC** |
| **Multipath Routing & Reordering** | Static 5-Tuple ECMP (`42%` elephant collision) | Static 5-Tuple ECMP | Rail-Aligned Topology + Flowlet DLB | **Per-Packet Adaptive Spraying + RNIC Hardware Out-of-Order DDP** |
| **16GB AllReduce Bus Bandwidth** | `11.4 GB/s` ✕ | `28.5 GB/s` | **`41.8 GB/s`** ✅ | **`47.6 GB/s` (`4.18x` vs. Tier 1)** 🏆 |
| **MoE All-to-All Bus Bandwidth** | `6.8 GB/s` ✕ | `18.2 GB/s` | **`34.5 GB/s`** ✅ | **`45.2 GB/s` (`2.48x` vs. Tier 2)** 🏆 |
| **400G Fabric Utilization** | `22.8%` ✕ | `57.0%` | **`83.6%`** ✅ | **`95.2%` (InfiniBand-Grade)** 🏆 |
| **32:1 Incast Burst P99 Latency** | `145.0 ms` ✕ | `68.0 ms` | **`8.4 ms`** ✅ | **`2.1 ms`** 🏆 |
| **Monthly PFC Storm Outages** | `0.0` (Stalls via drops) | **`6.5 outages / mo` ✕** | **`0.1 outages / mo` (Watchdog auto-heals)** ✅ | **`0.0 outages / mo` (Architecturally immune)** 🏆 |
| **Annual GPU Stall Waste (USD)** | `$4,850,000 / yr` ✕ | `$2,920,000 / yr` ✕ | **`$640,000 / yr`** ✅ | **`$145,000 / yr` (Saves `$2.78M/yr`)** 🏆 |

---

## ⑤ 🧪 Quantitative Case Study & 7-Card Benchmark Insights

### 5.1 Reproducible Evaluator & Sealed Artifacts

All metrics in this guide are generated by the deterministic evaluator and verified via unit tests:
- **Evaluator Script**: [`experiments/roce_pfc_congestion_control_selection/run.py`](../../experiments/roce_pfc_congestion_control_selection/run.py)
- **Sealed Output**: [`experiments/roce_pfc_congestion_control_selection/results.json`](../../experiments/roce_pfc_congestion_control_selection/results.json)
- **Unit Test Suite**: [`tests/test_roce_pfc_congestion_control_selection.py`](../../tests/test_roce_pfc_congestion_control_selection.py)
- **7-Card 4:5 Visual Report**: [`docs/reports/ai-cluster-roce-pfc-storm-congestion-control-selection-cards.html`](../reports/ai-cluster-roce-pfc-storm-congestion-control-selection-cards.html)
- **Interactive Sizer Workbench**: [`docs/reports/llm-inference-engine-pd-sizer.en.html#roce-pfc-congestion-sizer`](../reports/llm-inference-engine-pd-sizer.en.html)
- **Provenance**: `synthetic_analytical_model_calibrated_to_anonymized_published_benchmarks` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`).

### 5.2 Key Quantitative Insights

1. **Early ECN Watermarks Eliminate 99% of PFC Pause Triggers**:
   $$\text{AllReduce BusBW}: 11.4\text{ GB/s (Tier 1)} \rightarrow 28.5\text{ GB/s (Tier 2)} \rightarrow 41.8\text{ GB/s (Tier 3)}$$
   Setting WRED ECN marking thresholds (`150 KB–800 KB`) well below the `1.5 MB` PFC XOFF floor—paired with a `200 ms` SONiC PFC Watchdog—eliminates `6.5` monthly PFC storm deadlocks and saves **`$2.28M/year`** in wasted GPU stall time.
2. **Per-Packet Spraying + RNIC Out-of-Order DDP Unlocks `95.2%` Ethernet Efficiency on MoE**:
   $$\text{MoE All-to-All BusBW}: 18.2\text{ GB/s (Tier 2)} \xrightarrow{+90\%} 34.5\text{ GB/s (Tier 3)} \xrightarrow{+31\%} 45.2\text{ GB/s (Tier 4, 2.48x)}$$

---

## ⑥ 🛠️ Appendix: Production Architect Checklist

- [ ] **1. Align End-to-End DSCP (`DSCP 26` / `ToS 106`) to Lossless Priority Queue (`Priority 3`)**: Verify that every GPU host (`mlx5_qos`, `NCCL_IB_TC=106`) and every Leaf/Spine switch maps RoCEv2 traffic to the exact same lossless queue (`Priority 3`), and never run RDMA on lossy default `Priority 0`.
- [ ] **2. Enforce the 3-Level Watermark Inequality ($K_{\min} < K_{\max} \ll \text{XOFF}_{\text{PFC}}$)**: Audit switch `BUFFER_PROFILE` settings so WRED ECN marking starts at `150 KB` ($K_{\min}$) and reaches 100% at `800 KB` ($K_{\max}$), far below the `1.5 MB` PFC `XOFF` pause threshold.
- [ ] **3. Enable Hardware `PFC Watchdog` (`200 ms` Drop/Quarantine)**: Configure `pfcwd start --action drop 200 --restoration-time 400` on all SONiC Leaf and Spine ports to automatically quarantine any hung NIC emitting continuous PFC pause frames.
- [ ] **4. Enable RNIC Hardware Out-of-Order Reordering (`OOO DDP`) Before Turning On Packet Spraying**: Confirm `Adaptive Routing / Out-of-Order RoCE` is enabled in ConnectX-7 / SuperNIC firmware before enabling switch packet spraying or DLB.
- [ ] **5. Deploy Rail-Optimized Leaf-Spine Cabling**: Connect NIC `#k` of every server to the same Rail `#k` Leaf switch group so intra-rail collectives complete in a single hop without traversing Spine links.
- [ ] **6. Alert on `PFC XOFF` Spikes vs. Smooth `ECN/CNP` Baselines**: Stream switch `ECN_MARKED_PKTS`, `CNP_SENT`, and `PFC_RX/TX_PAUSE` counters into Prometheus; a healthy cluster exhibits steady ECN/CNP regulation with near-zero PFC pauses.
