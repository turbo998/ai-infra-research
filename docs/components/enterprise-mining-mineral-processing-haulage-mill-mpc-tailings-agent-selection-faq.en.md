# Enterprise Mining, Mineral Processing & Autonomous Haulage (`AHS`), Geometallurgical Ore-Blending `MILP`, SAG/Ball Mill Grinding `MPC` & Tailings Storage Facility (`GISTM InSAR + VWP`) Safety Agent Architecture Selection Guide: Fixed Truck-Shovel Locking + Single-Loop SAG PID vs. Prompt-Only Dispatch/Mill Log LLM vs. Shortest-Path GPS Dispatch + Uncoupled Mill ML vs. Shovel `XRF` Geomet Blending `MILP` + 3D Fragmentation Feed-Forward `SAG MPC` + `GISTM` Tailings Interlock Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-mining-mineral-processing-haulage-mill-mpc-tailings-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Chief Mine & Concentrator Metallurgists, Autonomous Haulage System (`Caterpillar Command / Komatsu FrontRunner`) Leads, and Tailings Storage Facility (`GISTM`) Engineers-of-Record.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_mining_mineral_processing_haulage_mill_mpc_tailings_agent_selection/results.json`](../../experiments/enterprise_mining_mineral_processing_haulage_mill_mpc_tailings_agent_selection/results.json).

![Enterprise Mining AHS Haulage Blending, SAG Mill MPC & Tailings InSAR Safety Agent Summary](../../assets/svg/enterprise-mining-mineral-processing-haulage-mill-mpc-tailings-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating a **250,000 tonnes/day open-pit and underground mining complex with `140` 300-tonne haul trucks (`AHS` / manned), `4` SAG/Ball mill comminution and froth-flotation lines, and `320` tailings storage facility (`TSF`) monitoring nodes** exposes three high-loss Mine-to-Mill failure modes:
1. **Shortest-Path GPS Truck Dispatch (Tier 3) Dumps Unblended High-Clay & High-Hardness Ore Into the Concentrator**: When Shovel #4 hits a `13.2%` smectite clay pocket, shortest-distance dispatch sends 6 consecutive 300-tonne trucks to the primary crusher—blinding vibrating screens and poisoning flotation bubble recovery (`6.4%–14.8%` grinding energy and metal recovery loss)!
2. **Single-Loop SAG Mill Tonnage PIDs (Tier 1) Suffer `16.4` Volumetric "Belly-Out" Overloads per Month**: When coarse blast fragmentation (`F80` jumping from `95mm` to `165mm`) enters the SAG mill, power draw counter-intuitively drops once charge fill exceeds the `~34%` turnover peak—tricking a conventional PID loop into adding *more* feed until the mill stalls for a 6-hour grind-out!
3. **Monthly Manual Tailings Dam Walks Miss Rapid Pore-Water Pressure (`B-bar >= 0.40`) Spikes After Heavy Rain (`3.8%` Alarm Escape)**.

Evaluated across a **250,000 t/d mining complex, 140 haul trucks, and 4 SAG/flotation lines**, the four architectural tiers compare as follows:

| Architectural Tier | Core Stack Combination | SAG Mill & Haulage Stalls / Mo | Truck Queue & Shovel Idle Waste | Grinding Energy & Recovery Loss | Tailings Seepage / Slope Escape | Monthly Mine-to-Mill Loss TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Fixed Truck-Shovel + Single-Loop SAG PID** | Locked shovel assignments + fixed-t/h SAG PID + monthly TSF walks | `16.4 / mo` ✕ | `23.6%` ✕ *(queues & starvation)* | `14.8%` ✕ *(belly-out & sliming)* | `3.8%` ✕ | `$1,420,000 / mo` ✕ | ⛔ Frequent mill belly-outs & truck fuel waste |
| **🟨 Tier 2: Prompt-Only Dispatch & SCADA Log LLM** | Feeding shift radio logs & alarms into an LLM to tune feed/routes | `14.8 / mo` ✕ | `21.2%` ✕ | `16.2%` ✕ *(slurry pooling)* | `4.5%` ✕ | `$1,560,000 / mo` ✕ | ⛔ Violates comminution & flotation physics |
| **🟦 Tier 3: Shortest-Path GPS + Uncoupled Mill ML** | Travel-time-only truck dispatch + isolated mill regression ML | `5.6 / mo` | `9.8%` ✕ | `6.4%` ✕ *(clay/BWi shocks)* | `1.2%` ✕ | `$520,000 / mo` ✕ | ⚠️ Siloed pit dispatch starves concentrator |
| **🟩 Tier 4: Geomet Blending `AHS MILP` + `SAG MPC` + `GISTM InSAR` Agent** | **Shovel `XRF` Blending `MILP` + Belt 3D `F80` Feed-Forward Acoustic/Bearing `SAG MPC` + Froth `VLM` + `GISTM InSAR/VWP` Gate** | **`0.3 / mo` 🏆** *(-98.2%)* | **`1.8%` 🏆** *(-21.8 pts)* | **`1.2%` 🏆** *(+2.8% Cu/Au rec)* | **`0.0%` 🏆** *(Zero escape)* | **`$135,000 / mo` 🏆** *(-90.5%)* | 🌟 **Recommended Mine-to-Mill Architecture** |

> 💡 **Golden Rule for Mining & Mineral Processing AI Architects**: **"Never optimize haul-truck dispatch on GPS travel time alone—embed geometallurgical blending constraints (`<=4.5%` smectite clay, deleteriously capped arsenic, and smoothed Bond Work Index `BWi`) directly inside the rolling-horizon `AHS MILP`, feed conveyor 3D laser fragmentation (`F80`) forward into a multivariable `SAG Mill MPC` anchored by trunnion bearing pressure and shell acoustics, and hard-interlock tailings spigots against `GISTM` vibrating-wire piezometer (`B-bar >= 0.40`) and `InSAR` thresholds."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Real-World Mine-to-Mill Pain Points)

### Q1: How does Geometallurgical Ore-Blending `AHS MILP` stop Pit Dispatch from ruining Concentrator Flotation Recovery?
- **First-Principles Reality**:
  - Shortest-path truck dispatch (Tier 3) minimizes haul distance without regard to mineralogy—sending consecutive loads of high-smectite clay (`>12%`) or ultra-hard quartz porphyry (`BWi = 21.5 kWh/t`) to the primary crusher.
  - Tier 4 fuses **Shovel Hydraulic Specific Dig Energy (`kWh/t`) + Bucket `XRF/LIBS` Grade Sensors** into a 15-minute rolling-horizon **`AHS Blending MILP`** that enforces:
    $$\frac{\sum_{s} M_s \cdot \text{Clay}_s}{\sum_s M_s} \le 4.5\%, \quad \left|\frac{\sum_s M_s \cdot \text{BWi}_s}{\sum_s M_s} - \overline{\text{BWi}}_{\text{target}}\right| \le 1.2\text{ kWh/t}$$
  - By ratio-blending 1 truck from high-clay Shovel #4 with 3 trucks from hard-rock Shovel #2 while balancing crusher dump queues, Tier 4 cuts haulage idle waste from `23.6%` to **`1.8%`** and stabilizes downstream flotation.

### Q2: Why do Single-Loop SAG Mill PIDs suffer "Belly-Out" Overloads, and how does 3D Fragmentation Feed-Forward + Acoustic/Bearing `MPC` prevent them?
- **Overcoming the Non-Monotonic SAG Power-to-Load Curve**:
  - Beyond `~32–35%` volumetric fill, a SAG mill enters a "slurry pooling / belly-out" regime where motor power *decreases* as charge weight *increases*—causing a naive power PID to overfeed the mill until it stalls.
  - Tier 4 pairs **On-Belt 3D Laser Fragmentation `VLM` (`F80`)** feed-forward with **Trunnion Hydrostatic Bearing Pressure (true charge mass) + Shell Microphone Acoustic Toe-Angle** inside a multivariable `MPC` (plus **Froth Flotation Bubble-Velocity `VLM`** for reagent dosing), cutting mill stalls to **`0.3/month`** and grinding/recovery losses to **`1.2%`**.

### Q3: How does the `GISTM` Tailings Storage Facility (`TSF`) Interlock guarantee `0.0%` seepage and slope-stability alarm escape?
- **Satellite `InSAR` Deformation + Vibrating-Wire Piezometer (`VWP`) `B-bar` Hardware Interlock**:
  - Whenever heavy rainfall drives a sector's pore-water pressure ratio $\bar{B} = \Delta u / \Delta \sigma_v \ge 0.40$ or satellite `InSAR` detects `>3.0 mm/week` crest movement, Tier 4 executes a deterministic `GISTM TARP Level-3` PLC interlock: shutting active deposition spigots on that sector, diverting slurry to emergency cells, and alerting the Engineer-of-Record (`0.0%` escape).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Mine-to-Mill Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Mine-to-Mill Haulage, Comminution/Flotation & Tailings Safety<br/>(250,000 t/d ROM Ore + 140 300t Haul Trucks + 4 SAG Lines + 320 TSF Nodes)"] --> B{"Uses fixed shovel locking + single-loop SAG PID<br/>or prompt-only LLM dispatch/mill tuning?"}
    B -- "Yes (Fixed dispatch + single-loop PID)" --> C["⛔ SAG Belly-Outs & High Haulage Queues (Tier 1 / Tier 2)!<br/>16.4 mill stalls/mo, 23.6% truck idle waste & $1.42M/mo operational loss"]
    B -- "Uses GPS dispatch & ML" --> D{"Couples geometallurgical blending MILP, belt 3D fragmentation<br/>SAG MPC, froth VLM & GISTM InSAR/VWP interlocks?"}
    D -- "No (Shortest-path GPS dispatch without blending)" --> E["⚠️ Clay/Hardness Shocks Choke Concentrator (Tier 3)<br/>6.4% grinding/recovery loss & 9.8% truck queue waste"]
    D -- "Yes (Geomet MILP + SAG MPC + GISTM InSAR Gate)" --> F["🟩 Tier 4: Geomet Blending MILP + 3D Fragmentation SAG MPC + GISTM Agent 🏆<br/>(0.3 Stalls/Mo | 1.8% Truck Waste | 1.2% Recovery Loss | 0.0% TSF Escape)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Geometallurgical `AHS MILP` + 3D Fragmentation `SAG MPC` + `GISTM` Tailings Topology

```mermaid
flowchart LR
    subgraph Pit["1. Shovel Sensing & Geomet Blending MILP"]
        Shovel["Shovel Dig-Energy (BWi)<br/>+ Bucket XRF Clay/Arsenic"] --> MILP["🚜 140-Truck AHS Blending MILP<br/>Caps Smectite Clay <=4.5% & Smooths BWi<br/>(1.8% Queue/Starvation Waste)"]
    end

    subgraph Mill["2. Concentrator 3D F80 SAG MPC & Froth VLM"]
        MILP --> Belt["📷 Belt 3D Laser F80 VLM<br/>(Dust Fallback to Crusher Current)"]
        Belt --> MPC["⚙️ Bearing-Pressure + Acoustic SAG MPC<br/>+ Froth Bubble Velocity/D50 VLM<br/>(0.3 Stalls/Mo | 1.2% Loss)"]
    end

    subgraph TSF["3. GISTM Tailings InSAR + VWP Interlock"]
        MPC --> Dam["🛰️ Satellite InSAR + 320 VWP Piezometers<br/>B-bar >= 0.40 Auto-Diverts Spigots<br/>(0.0% Tailings Alarm Escape)"]
    end
```

### 3.3 End-to-End Sequence: Blending a `13.2%` Clay Pocket & Preventing an `F80=165mm` SAG Mill Belly-Out

```mermaid
sequenceDiagram
    autonumber
    participant Pit as Shovel #4 (XRF/Dig-Energy) & 140 AHS Trucks
    participant MILP as Geometallurgical Blending AHS MILP
    participant MPC as Belt 3D VLM + SAG Mill & Flotation MPC
    participant TSF as GISTM Tailings InSAR/VWP Gate

    Pit->>MILP: Shovel #4 XRF detects smectite clay spike at 13.2% (unblended dump would blind screens & poison flotation)
    MILP->>Pit: ⚡ Re-solve 15-min AHS MILP: blend 1 truck from Shovel #4 per 3 trucks from Shovel #2 -> crusher clay locked at 4.1% (<=4.5%)
    Pit->>MPC: Belt 3D laser camera flags coarse blast rock (F80 jumps from 95mm to 165mm) 4 minutes upstream of SAG Mill #1
    MPC->>MPC: Fuse trunnion bearing pressure + acoustic toe-angle -> trim fresh feed -8% & raise VSD speed +0.6 rpm (zero belly-out!)
    TSF->>TSF: Heavy rain pushes Sector C piezometer ratio B-bar to 0.44 (>=0.40 limit) + InSAR crest shift 3.8 mm/wk
    TSF->>MPC: 🚨 Trigger GISTM TARP-3 PLC interlock: close Sector C spigots, divert slurry to Cell B & page Engineer-of-Record!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Fixed Shovel + SAG PID | 🟨 Tier 2: Prompt Dispatch/Mill LLM | 🟦 Tier 3: Shortest-Path GPS + ML | 🟩 Tier 4: Geomet `MILP` + `SAG MPC` + `InSAR` |
|---|---|---|---|---|
| **SAG Mill & Haulage Stalls / Mo** | `16.4 / mo` ✕ | `14.8 / mo` ✕ | `5.6 / mo` | 🏆 **`0.3 / mo` (`-98.2%` vs. baseline)** |
| **Truck Queue & Shovel Idle Waste** | `23.6%` ✕ (queues & starvation) | `21.2%` ✕ | `9.8%` | 🏆 **`1.8%` (rolling-horizon geometallurgical `AHS MILP`)** |
| **Grinding Energy & Recovery Loss** | `14.8%` ✕ (belly-out & assay lag) | `16.2%` ✕ (slurry pooling) | `6.4%` ✕ (clay/hardness shocks) | 🏆 **`1.2%` (3D `F80` feed-forward `SAG MPC` + Froth `VLM`)** |
| **Tailings Seepage / Slope Escape** | `3.8%` ✕ (monthly manual walk) | `4.5%` ✕ | `1.2%` ✕ | 🏆 **`0.0%` (`GISTM InSAR` + `VWP B-bar >= 0.40` gate)** |
| **Monthly Mine-to-Mill Loss TCO** | `$1,420,000 / mo` | `$1,560,000 / mo` | `$520,000 / mo` | **`$135,000 / mo` 🏆 (-90.5%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Mining AHS Haulage, SAG Mill MPC & Tailings Safety Sizer](../reports/enterprise-mining-mineral-processing-haulage-mill-mpc-tailings-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-mining-mineral-processing-haulage-mill-mpc-tailings-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_mining_mineral_processing_haulage_mill_mpc_tailings_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_mining_mineral_processing_haulage_mill_mpc_tailings_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Embed Geometallurgical Blending Constraints (`Smectite Clay <= 4.5%`, Arsenic Caps & Bond Work Index `BWi` Smoothing) Inside the `AHS` Dispatch `MILP`**: Never dispatch haul trucks on shortest GPS travel distance alone.
- [ ] **Control `SAG` Mills via Multivariable `MPC` Fusing Belt 3D Fragmentation (`F80`), Trunnion Bearing Pressure & Shell Acoustic Toe-Angle**: Prevent non-monotonic power-curve "belly-out" overloads.
- [ ] **Configure Dust-Obscuration Telemetry Fallbacks on Primary Crusher Belts**: Fall back automatically to crusher mantle current and shovel dipper dig-energy (`kWh/t`) when dust obscures optical `F80` cameras.
- [ ] **Close the Loop on Rougher/Cleaner Flotation via Froth Bubble-Velocity and Size (`D50`) `VLMs`**: Adjust collector/frother dosing and pulp levels continuously between 4-hour lab fire assays.
- [ ] **Enforce Hardware PLC Spigot Interlocks on Tailings Storage Facilities (`GISTM TARP`)**: Divert slurry deposition automatically whenever vibrating-wire piezometers breach $\bar{B} \ge 0.40$ or satellite `InSAR` flags accelerating crest deformation.
