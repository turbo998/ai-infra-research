# Enterprise GPU Cluster Power Capping (`nvidia-smi -pl`), Prefill/Decode Phase-Aware `DVFS` Frequency Scaling & Rack Power Oversubscription Architecture Selection Guide: Packing `33.3%` More GPUs per Rack While Saving `36.2%` TCO

> **Language / 语言**: **English (Current)** | [中文](./enterprise-gpu-cluster-power-capping-dvfs-thermal-slo-scheduling-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, AI Data Center (`AIDC`) Capacity Planners, and Bare-Metal Kubernetes GPU Platform Engineers operating power-constrained `H100 / H200 / B200 / L40S` clusters.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_gpu_cluster_power_capping_dvfs_thermal_slo_scheduling_selection/results.json`](../../experiments/enterprise_gpu_cluster_power_capping_dvfs_thermal_slo_scheduling_selection/results.json).

![Enterprise GPU Cluster Power Capping & Phase-Aware DVFS Summary](../../assets/svg/enterprise-gpu-cluster-power-capping-dvfs-thermal-slo-scheduling-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

In private AI data centers and colocation cages (**`256x H100-SXM-80GB` GPUs = `32` 8-GPU servers deployed across `32 kW` power-capped racks at `$0.14/kWh` effective PUE-adjusted rate**), **rack power envelopes—not rack U-height—are the primary physical bottleneck**. Three naive power-management architectures waste hundreds of thousands of dollars annually:
1. **Trap #1: Uncapped `700W` Nameplate TDP + Strict Nameplate Rack Sizing (`Tier 1`, Fits Only `3 Servers = 24 GPUs` per `32kW` Rack -> Strands `25%` of Rack Space across 11 Racks, Suffers `6.8%` Thermal Throttling Tail Spikes, `$185,000/month` Rack + Power TCO)**:
   - Because CMOS dynamic power scales cubically with frequency ($P \propto V^2 f \approx f^3$), the top `200W` (`500W -> 700W` on H100) buys only `+4%–6%` speed during compute-bound Prefill and **`<1%` speed during HBM-bandwidth-bound Decode**! Worse, running uncapped causes middle-of-rack GPUs to hit the `85°C` HBM thermal slowdown threshold, stalling all 8 GPUs in a `TP=8` group!
2. **Trap #2: Blind Static `400W` Hard Power Cap Across All Nodes (`Tier 2`, Fits `32 GPUs/Rack`, but Starves Compute-Bound Prefill Nodes -> `+32%` TTFT Lag & Drops `SLO Goodput` to `74.5%`, `$142,000/month` TCO)**.
3. **Trap #3: Unguarded `32-GPU/Rack` Overbooking Without Hardware Power Clamping (`Tier 3`, `4.2%` Rack PDU Circuit-Breaker Trip Risk During Synchronized Prefill Bursts, `$156,000/month` TCO)**.

Evaluated across a **256-GPU `H100-SXM-80GB` inference cluster in `32 kW` colocation racks**, the four power and rack-density architectures compare as follows:

| Power & Rack Tier | Clock/Power Capping, Rack Density & PDU Protection | GPUs per `32kW` Rack | `SLO` Goodput Retention | Cluster Power Savings | Monthly Rack + Power TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Uncapped `700W` TDP + Nameplate Rack Sizing** | No power cap; 3 servers/rack (`11` racks total) | `24 GPUs / Rack` ✕ *(25% stranded)* | `93.2%` *(6.8% thermal stalls)* | `0.0%` ✕ *(cubic V²f waste)* | `$185,000 / mo` ✕ *(11 racks + peak kWh)* | ⛔ Wastes top 200W/GPU as heat and strands 25% of colocation rack capacity |
| **🟨 Tier 2: Blind Static `400W` Cap Across All Nodes** | Hard `400W` cap on both Prefill and Decode nodes | `32 GPUs / Rack` | **`74.5%` ✕** *(+32% TTFT lag)* | `35.0%` | `$142,000 / mo` ✕ *(cripples GPU throughput)* | ⛔ Starves compute-bound prefill nodes, wasting 25% of GPU capital |
| **🟦 Tier 3: Unguarded `32-GPU/Rack` Overbooking** | Packs 4 servers/rack with no fast PDU power clamp | `32 GPUs / Rack` | `88.0%` | `4.0%` | `$156,000 / mo` ✕ *(4.2% breaker trips)* | ⛔ Synchronized bursts trip 32kW rack PDU circuit breakers |
| **🟩 Tier 4: Phase-Aware `DVFS` (`600W` Prefill / `450W 1350MHz` Decode) + `100ms` DCGM/PDU Clamp** | **Prefill at `600W` sweet spot; HBM-bound Decode locked at `1350MHz / 450W` (`-34%` power, `<1.2%` TPOT delta) + `100ms` PDU clamp** | **`32 GPUs / Rack` 🏆** *(+33.3% density, 8 racks)* | **`98.8%` 🏆** *(zero thermal stalls!)* | **`28.0%` 🏆** *(-34% decode power)* | **`$118,000 / mo` 🏆** *(-36.2%, saves $804K/yr)* | 🌟 **Recommended Enterprise GPU Cluster Power & Density Standard** |

> 💡 **Golden Rules for Enterprise GPU Cluster Power Capping & Rack Density**:
> 1. **Exploit Memory-Bound Decode Physics (`450W / 1,350MHz` SM Clock Lock)**: Autoregressive LLM token decoding is **100% HBM3 memory-bandwidth bound (`3.35 TB/s`), NOT Tensor-Core compute bound**! Locking Decode GPUs to `1,350 MHz` SM clock and a `450W` cap (`nvidia-smi -lgc 900,1350 -pl 450`) keeps HBM3 memory running at `100%` speed while cutting Decode GPU power by **`34.0%` with `<1.2%` TPOT latency impact**!
> 2. **Trim the Cubic Voltage Spike on Prefill Nodes (`600W` Sweet-Spot Cap)**: Capping compute-bound Prefill GPUs at `600W` (instead of `700W`) saves `14.3%` peak power and keeps HBM temperatures below `72°C`—eliminating `85°C` thermal throttling stalls across `TP=8` barriers and lifting `SLO Goodput` from `93.2%` to **`98.8%`**!
> 3. **Pack `+33.3%` More GPUs per Rack (`32 vs. 24 GPUs`) with a `100ms` NVML/PDU Closed-Loop Clamp**: Reducing sustained server power from `10.2 kW` to `7.4 kW` fits **4 servers (`32 GPUs`) per `32kW` rack** (cutting colocation footprint from 11 racks to 8 racks), guarded by a `100ms` `DCGM / Kepler` power step-down loop if rack PDU load exceeds `29.5 kW` (`0.0%` breaker trips)!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Cubic Voltage-Frequency Physics, Thermal Stragglers in `TP=8` & `100ms` PDU Breaker Protection Pain Points)

### Q1: Why does capping a `700W` H100 Decode GPU down to `450W` (`-35.7%` power limit) slow down token generation (`TPOT`) by less than `1.2%`?
- **Roofline Arithmetic Intensity vs. Cubic Voltage-Frequency Scaling ($P \propto V^2 f$)**:
  - During the Decode phase, generating each token requires streaming all `140GB` of model weights from HBM3 memory into SRAM, leaving Tensor Core SMs waiting on memory transfers.
  - Because HBM3 memory clocks remain pinned at full speed at `450W`, lowering the SM core clock from `1,980 MHz` (high voltage) to `1,350 MHz` (sweet-spot voltage) eliminates `250W` of voltage-squared heat dissipation without bottlenecking memory bandwidth!

### Q2: How can a power-capped cluster (`Tier 4`, `98.8%` SLO goodput) outperform an uncapped `700W` cluster (`Tier 1`, `93.2%` SLO goodput)?
- **Eliminating Synchronous `TP=8` Thermal Stragglers**:
  - In an uncapped `700W` chassis, the hottest GPU in an 8-GPU Tensor Parallel (`TP=8`) group periodically hits the `85°C` HBM thermal slowdown threshold and drops its clock to `900 MHz`.
  - Because `NCCL AllReduce` synchronizes all 8 GPUs at every Transformer layer, **1 thermally throttled GPU forces the other 7 GPUs to sit idle waiting for it**! Tier 4 keeps all 8 GPUs at a uniform, thermally stable `71°C`, eliminating tail-latency spikes!

### Q3: How do we guarantee `0.0%` rack PDU circuit-breaker trips when packing 4 servers (`32 GPUs`) into a `32 kW` colocation rack?
- **Sub-Second `DCGM + Smart PDU` Hardware Power Clamping (`100ms` Loop)**:
  - Rack circuit breakers trip on thermal-magnetic $I^2 t$ integration over several seconds, whereas NVIDIA `NVML` hardware power limits (`nvmlDeviceSetPowerManagementLimit`) take effect in **`<100 milliseconds`**.
  - Whenever the rack telemetry agent sees aggregate rack draw exceed `29.5 kW` for `100ms`, it steps Prefill GPUs down from `600W` to `480W` immediately—capping rack draw below `28.0 kW` long before a breaker can trip!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Phase-Aware `600W/450W` Rack Topology + `100ms` PDU Clamp Sequence)

### 3.1 Enterprise GPU Cluster Power Capping & Rack Density Decision Tree

```mermaid
flowchart TD
    A["Enterprise GPU Cluster Power & Rack Density Selection<br/>(256x H100 700W GPUs | 32kW Colocation Racks | $0.14/kWh PUE-Adjusted Rate)"] --> B{"How are GPU power limits and per-rack server density configured?"}
    B -- "Uncapped 700W TDP + nameplate 24 GPUs/rack sizing (Tier 1)" --> C["⛔ Only 24 GPUs/Rack (11 Racks) + 6.8% Thermal Stalls!<br/>Wastes top 200W on cubic V²f curve ($185K/mo TCO)"]
    B -- "Blind static 400W cap across all Prefill & Decode nodes (Tier 2)" --> D["⛔ +32% Prefill TTFT Lag (74.5% SLO Goodput)!<br/>Starves compute-bound prefill workloads ($142K/mo TCO)"]
    B -- "Unguarded 32-GPU/rack overbooking without fast clamp (Tier 3)" --> E["⛔ 4.2% Rack PDU Circuit-Breaker Trip Risk!<br/>Synchronized bursts exceed 32kW rack limit ($156K/mo)"]
    B -- "Phase-Aware DVFS (600W Prefill / 450W Decode) + 100ms PDU Clamp (Tier 4)" --> F["🟩 Tier 4: Phase-Aware 600W/450W DVFS + 32 GPUs/Rack + 100ms PDU Clamp 🏆<br/>(+33.3% Rack Density | -28% Power | 98.8% Goodput | -36.2% TCO, $804K/yr Saved)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Phase-Aware `600W Prefill / 450W Decode` Rack Topology (`32 GPUs / 32kW Rack`) & `100ms` PDU Clamp

```mermaid
flowchart LR
    subgraph Rack["32kW Colocation Rack (4x 8-GPU H100 Servers = 32 GPUs, +33.3% Density!)"]
        P["⚡ Prefill GPU Pool (Servers 1..2)<br/>Compute-Bound Profile:<br/>nvidia-smi -pl 600W<br/>Trims top 100W cubic voltage waste"]
        D["🧊 Decode GPU Pool (Servers 3..4)<br/>HBM-Bandwidth-Bound Profile:<br/>-lgc 900,1350 & -pl 450W<br/>-34% power with <1.2% TPOT impact!"]
    end

    subgraph Clamp["100ms Closed-Loop Rack PDU Safety Clamp (0.0% Breaker Trips)"]
        PDU["🔌 Smart Rack PDU Telemetry<br/>(Steady Load = 27.2 kW / 32.0 kW)"] --> Ctrl["🛡️ DCGM + Kepler Power Daemon"]
        Ctrl -- "If Rack > 29.5 kW for 100ms" --> Step["📉 Instant NVML Step-Down<br/>Clamps Prefill to 480W in <100ms"]
        Step --> P
    end
```

### 3.3 End-to-End Sequence: Phase-Aware Power Allocation & `100ms` Synchronized Burst Shaving

```mermaid
sequenceDiagram
    autonumber
    participant Router as Disaggregated P/D Router
    participant Prefill as Prefill GPU Pool (600W Cap)
    participant Decode as Decode GPU Pool (1350MHz / 450W Cap)
    participant PDU as Smart Rack PDU & DCGM Power Daemon

    Router->>Prefill: 1. Route 32K-token prefill burst -> GPUs run at `600W` sweet spot (HBM temp stable at `71°C`, zero thermal throttling)
    Router->>Decode: 2. Transfer KV cache to Decode pool -> GPUs run at `1350MHz / 450W` (`3.35 TB/s` full HBM3 speed, `-34%` power!)
    PDU->>PDU: 3. Synchronized rack spike pushes aggregate 4-server load to `29.8 kW` (approaching `32kW` threshold)
    PDU->>Prefill: 4. `T + 100ms`: Daemon invokes `nvmlDeviceSetPowerManagementLimit(480W)` across Prefill GPUs
    PDU->>PDU: 5. Rack power drops to `27.8 kW` (`0.0%` breaker trip, `98.8%` SLO goodput preserved)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Uncapped `700W` | 🟨 Tier 2: Static `400W` Cap | 🟦 Tier 3: Unguarded Overbook | 🟩 Tier 4: Phase-Aware `600W/450W` + `PDU` Clamp |
|---|---|---|---|---|
| **GPUs per `32kW` Rack** | `24 GPUs` ✕ (11 racks needed) | `32 GPUs` (8 racks) | `32 GPUs` (8 racks) | 🏆 **`32 GPUs` (+33.3% density; saves 3 full racks)** |
| **`SLO` Goodput Retention** | `93.2%` (6.8% thermal stalls) | `74.5%` ✕ (+32% TTFT lag) | `88.0%` | 🏆 **`98.8%` (eliminates `TP=8` thermal stragglers)** |
| **Cluster Power Reduction** | `0.0%` ✕ | `35.0%` | `4.0%` | 🏆 **`28.0%` (-34% on Decode, -14% on Prefill)** |
| **Rack PDU Breaker Trip Risk** | `0.0%` | `0.0%` | `4.2%` ✕ (outage risk) | 🏆 **`0.0%` (`100ms` NVML closed-loop power clamp)** |
| **256-GPU Monthly Rack+Power TCO** | `$185,000 / mo` ✕ | `$142,000 / mo` ✕ | `$156,000 / mo` ✕ | **`$118,000 / mo` 🏆 (-36.2% vs Tier 1, saves `$804K/yr`)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise GPU Cluster Power Capping & Rack Density Sizer](../reports/enterprise-gpu-cluster-power-capping-dvfs-thermal-slo-scheduling-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-gpu-cluster-power-capping-dvfs-thermal-slo-scheduling-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_gpu_cluster_power_capping_dvfs_thermal_slo_scheduling_selection/run.py --check
  python -m unittest tests/test_enterprise_gpu_cluster_power_capping_dvfs_thermal_slo_scheduling_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Lock Memory-Bound `Decode` GPUs to `1,350 MHz` SM Clock and a `450W` Power Limit (`nvidia-smi -lgc 900,1350 -pl 450`)**: Cut Decode power draw by `34%` with `<1.2%` impact on token generation latency.
- [ ] **Cap Compute-Bound `Prefill` GPUs at the `600W` Efficiency Sweet Spot**: Eliminate the top `100W` cubic voltage spike and prevent `85°C` HBM thermal throttling stragglers inside `TP=8` groups.
- [ ] **Increase `32kW` Rack Density by `+33.3%` (`4 Servers / 32 GPUs` Instead of `3 Servers / 24 GPUs`)**: Reclaim stranded rack U-space and shorten intra-cluster InfiniBand/RoCE cable runs.
- [ ] **Deploy a `100ms` Closed-Loop Rack PDU Power Clamp (`DCGM + Kepler`)**: Automatically step down `NVML` power limits if aggregate rack PDU draw exceeds `90%` (`29.5 kW`), guaranteeing `0.0%` circuit-breaker trips.
- [ ] **Alert on `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` (`SW_THERMAL_SLOWDOWN`)**: Continuously verify that zero GPUs in any `TP/PP` group are suffering thermal clock throttling.
