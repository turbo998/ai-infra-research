# Enterprise Oil, Gas & Pipeline Midstream Leak Detection, Pigging Inline-Inspection (`ILI` MFL/Ultrasonic) Corrosion Growth, `API 579` Fitness-for-Service (`FFS`) & Drilling Lost-Circulation/Stuck-Pipe Prevention Agent Architecture Selection Guide: Static 5% Line-Balance + Odometer Excel vs. Prompt-Only SCADA/Mud-Log LLM vs. Uncompensated Flow ML vs. `API 1130` Compressible `RTTM` + Negative-Pressure-Wave `ToF` + Girth-Weld `DTW ASME B31G` + Soft-String `Torque-Drag` Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-pipeline-midstream-leak-ili-corrosion-drilling-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, VPs of Midstream Pipeline Integrity (`API 1130 / PHMSA`), Real-Time Drilling Operations Centers (`WITSML`), and Upstream/Midstream `SCADA` Engineering Teams.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_pipeline_midstream_leak_ili_corrosion_drilling_agent_selection/results.json`](../../experiments/enterprise_pipeline_midstream_leak_ili_corrosion_drilling_agent_selection/results.json).

![Enterprise Pipeline Midstream RTTM Leak, ILI Corrosion & Drilling Stuck-Pipe Agent Summary](../../assets/svg/enterprise-pipeline-midstream-leak-ili-corrosion-drilling-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating a **6,800 km midstream transmission pipeline network (`140,000` tracked Inline Inspection `ILI` smart-pig anomalies at `$45,000` per excavation dig)** alongside **14 directional drilling rigs** exposes three high-consequence engineering failure modes:
1. **Uncompensated Compressible Linepack Transients Blind Static Alarms to `0.8%` Pinhole Leaks (180-Minute Lag) or Trigger `19.5%` False Shut-Ins**: When a pump station starts up or daytime solar heating warms above-ground pipe segments by `+18°C`, fluid compressibility and thermal expansion alter the pipeline's internal mass inventory (**Linepack $dM/dt$**). Static 5% volume-balance rules (Tier 1) miss a `0.8%` pinhole leak or illegal tap for **3 hours (`180 min`)**, while uncompensated ML models on raw flowmeters (Tier 3) mistake normal pump startups for ruptures—triggering false emergency shut-ins (`$185K` downtime per event)!
2. **Smart-Pig Odometer Wheel Slip (`+4.5m`) Wastes `$45,000` per Misaligned Integrity Dig (`38.0%` Overkill)**: Comparing 2022 and 2026 Magnetic Flux Leakage (`MFL`) smart-pig runs by raw odometer distance in Excel (Tier 1) digs up healthy pipe 4.5 meters away from the actual corrosion pit cluster.
3. **Late Detection of Drillstring Pack-Offs Causes `64.0 Hours/Month` of Stuck-Pipe Non-Productive Time (`NPT`)**: Waiting for rotary torque to stall completely at `4,200m` measured depth results in `$1.9M` fishing/sidetrack jobs, while prompt-only LLMs (Tier 2) recommend dangerous mud-weight reductions below formation pore pressure!

Evaluated across a **6,800 km pipeline network, 140,000 ILI anomalies, and 14 directional drilling rigs**, the four architectural tiers compare as follows:

| Architectural Tier | Core Stack Combination | `0.8%` Pinhole Leak Detection Time | False Emergency Shut-In Rate | Unnecessary `$45K` Dig Overkill | Drilling Stuck-Pipe `NPT` | Monthly Shut-In, Dig & `NPT` Loss | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Static 5% Balance + Odometer Excel** | Static inlet-minus-outlet 5% alarm + linear smart-pig odometer matching in Excel | `180.0 min (3h)` ✕ | `19.5%` ✕ | `38.0%` ✕ *(4.5m slip)* | `64.0 hrs/mo` ✕ | `$980,000 / mo` ✕ | ⛔ Misses pinhole leaks & digs healthy pipe |
| **🟨 Tier 2: Prompt-Only SCADA & Mud-Log LLM** | Feeding SCADA pressure logs & drilling reports into an LLM for shut-in & mud advice | `95.0 min` ✕ | `24.0%` ✕ | `41.5%` ✕ | `58.0 hrs/mo` ✕ | `$1,140,000 / mo` ✕ | ⛔ Unsafe mud-weight drops risk well kicks |
| **🟦 Tier 3: Uncompensated Flow ML Only** | Time-series ML on flowmeters without RTTM linepack PDEs or DTW weld alignment | `28.0 min` | `8.2%` ✕ *(pump transient)* | `16.5%` ✕ | `26.0 hrs/mo` | `$460,000 / mo` ✕ | ⚠️ False alarms on normal pump startups |
| **🟩 Tier 4: `API 1130 RTTM` + `NPW ToF` + `DTW ASME B31G` + `Torque-Drag` Agent** | **Compressible `RTTM` PDEs + Acoustic `NPW` Time-of-Flight + Girth-Weld `DTW` `ASME B31G` Gate + Soft-String `Torque-Drag` & `PP-FG` Window** | **`1.8 min` 🏆** *(100× faster)* | **`0.3%` 🏆** *(-19.2 pts)* | **`1.4%` 🏆** *(-36.6 pts)* | **`3.5 hrs/mo` 🏆** *(-94.5%)* | **`$96,000 / mo` 🏆** *(-90.2%)* | 🌟 **Recommended Architecture** |

> 💡 **Golden Rule for Midstream Pipeline & Drilling AI Architects**: **"Never run leak-detection ML on raw inlet-minus-outlet flow without subtracting `API 1130` compressible linepack inventory ($dM/dt$) and coupling acoustic Negative-Pressure-Wave (`NPW`) time-of-flight (`1.8 min`, `±45m`), never align multi-run smart-pig corrosion data without Dynamic Time Warping (`DTW`) on girth-weld magnetic rings (`±2cm`) gated by `Modified ASME B31G` burst physics, and always bound drilling advisory inside soft-string `Torque & Drag` and Pore-Pressure/Fracture-Gradient (`PP-FG`) safety windows."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Real-World Pipeline & Drilling Pain Points)

### Q1: How does coupling `API 1130` Compressible `RTTM` with Acoustic Negative-Pressure-Wave (`NPW`) Time-of-Flight detect a `0.8%` pinhole leak in `1.8 minutes` (`±45m`) during a pump startup?
- **First-Principles Reality**:
  1. **Subtracting Transient Linepack Inventory ($dM/dt$)**: Tier 4 solves the 1D compressible continuity, momentum, and equation-of-state PDEs along the pipeline profile:
     $$Q_{\text{leak}}(t) = Q_{\text{in}}(t) - Q_{\text{out}}(t) - \int_0^L \left( \frac{A}{c^2}\frac{\partial P}{\partial t} - \alpha \rho A \frac{\partial T}{\partial t} \right) dx$$
     preventing pump startups and `+18°C` solar thermal expansion from triggering false shut-ins (`0.3%` false alarm rate).
  2. **Acoustic Negative-Pressure-Wave (`NPW`) Cross-Correlation**: When a `0.8%` pinhole rupture or illegal tap opens at `KP-142.6`, rarefaction waves propagate upstream and downstream at acoustic velocity $c \approx 1,150\text{ m/s}$. Cross-correlating arrival timestamps ($\Delta t = 41.2\text{ms}$) across adjacent pressure transmitters pinpoints `KP-142.62 ± 45m` in **`1.8 minutes`**.

### Q2: How does Girth-Weld Dynamic Time Warping (`DTW`) + `Modified ASME B31G` slash unnecessary `$45,000` integrity digs from `38.0%` to `1.4%`?
- **Defeating Smart-Pig Odometer Slip & Velocity Flux Distortion**:
  - Smart-pig mechanical odometer wheels slip in paraffin/wax deposits, causing `+4.5m` cumulative drift between 2022 and 2026 runs. Tier 4 uses the immutable magnetic spikes of **32,000 pipe girth welds** (spaced every ~12m) as landmarks, applying **Dynamic Time Warping (`DTW`)** to align multi-year corrosion pits within **`±2 cm`**.
  - It then applies velocity-induced eddy-current corrections and evaluates the **`Modified ASME B31G / API 579` Folias bulging factor ($M$)** burst-pressure equation—scheduling `$45,000` excavations strictly where projected Safe Working Pressure drops below $1.25 \times \text{MAOP}$ (`1.4%` overkill).

### Q3: How does the Soft-String `Torque & Drag` Residual + `PP-FG` Window cut drilling stuck-pipe `NPT` by `94.5%`?
- **15-Minute Early Pack-Off Interception + Deterministic Well-Control Bounds**:
  - By comparing real-time `WITSML` pick-up hookload and surface torque against a physics-based **Soft-String Torque & Drag** model, Tier 4 detects friction-factor divergence (`0.24 -> 0.44 > 0.38`) and annular cuttings loading **15 minutes before the drillstring packs off**—automatically commanding the driller to halt ROP, reciprocate, and sweep a high-viscosity pill (`3.5 hrs/mo` NPT vs. `64.0 hrs/mo`).
  - During fractured-carbonate lost circulation (`-12 bbl`), Managed Pressure Drilling (`MPD`) choke adjustments are clamped strictly inside the **`[Pore Pressure + 150 psi, Fracture Gradient]`** window so an AI advisor can never induce an underbalanced gas kick.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + RTTM/ILI/Drilling Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Midstream Pipeline Integrity & Directional Drilling<br/>(6,800 km Network + 140K ILI Anomalies + 14 Drilling Rigs)"] --> B{"Uses static 5% volume balance + odometer Excel<br/>or prompt-only LLM SCADA/mud-log summaries?"}
    B -- "Yes (Static balance or prompt LLM)" --> C["⛔ Leak Blindness & Wasted Digs (Tier 1 / Tier 2)!<br/>180m pinhole leak lag, 38% wasted $45K digs & 64h/mo stuck-pipe NPT"]
    B -- "Uses flow & sensor ML models" --> D{"Couples API 1130 compressible RTTM + NPW ToF,<br/>DTW girth-weld ASME B31G & Torque-Drag PP-FG gates?"}
    D -- "No (Uncompensated flow ML only)" --> E["⚠️ Linepack False Shut-Ins (Tier 3)<br/>8.2% false shut-ins on pump startup & 16.5% dig overkill"]
    D -- "Yes (RTTM + NPW + DTW B31G + Torque-Drag)" --> F["🟩 Tier 4: API 1130 RTTM + NPW ToF + DTW ASME B31G + Torque-Drag Agent 🏆<br/>(1.8m Leak Detect | 0.3% False Shut-In | 1.4% Dig Overkill | 3.5h/mo NPT)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `API 1130 RTTM` + Girth-Weld `DTW ASME B31G` + Soft-String `Torque-Drag` Topology

```mermaid
flowchart LR
    subgraph Leak["1. API 1130 RTTM & Acoustic NPW Plane"]
        SCADA["High-Rate Flow/Pressure/Temp<br/>+ Acoustic NPW Transmitters"] --> RTTM["🌊 Compressible Linepack PDE (dM/dt)<br/>+ 1,150 m/s NPW Cross-Correlation<br/>Pinpoints KP-142.6 (±45m in 1.8m)"]
    end

    subgraph Pig["2. DTW Smart-Pig & ASME B31G Plane"]
        MFL["2022 & 2026 MFL Smart-Pig Runs<br/>(140,000 Flux Anomalies)"] --> DTW["🧲 32,000 Girth-Weld DTW Aligner (±2cm)<br/>+ Velocity Flux Calibration & B31G<br/>(1.4% Excavation Overkill)"]
    end

    subgraph Drill["3. Soft-String Torque-Drag & PP-FG Gate"]
        Rig["14-Rig Live WITSML Streams<br/>(Hookload / Torque / SPP / Pits)"] --> TD["⚙️ Soft-String Torque & Drag Model<br/>15-Min Pre-Packoff Warning<br/>+ Hard PP-FG Well-Control Clamp"]
    end
```

### 3.3 End-to-End Sequence: Pinpointing a `KP-142.6` Pinhole Leak in 1.8m & Preventing a `4,200m` Stuck Pipe

```mermaid
sequenceDiagram
    autonumber
    participant Pipe as Pipeline SCADA & NPW Sensors
    participant RTTM as API 1130 RTTM + NPW Engine
    participant ILI as Girth-Weld DTW + ASME B31G Engine
    participant Drill as Torque-Drag + MPD PP-FG Gate

    Pipe->>RTTM: Pump station starts up (linepack dM/dt surge) while a 0.8% pinhole leak opens at KP-142.6
    RTTM->>RTTM: Subtract compressible linepack inventory & cross-correlate 41.2ms NPW acoustic arrival delta
    RTTM-->>Pipe: 🚨 Isolate pinhole leak at KP-142.62 ± 45m in 1.8 minutes (Act automatically) with zero false shut-in!
    ILI->>ILI: Align 2022 & 2026 MFL smart-pig runs across 32,000 girth welds via DTW (correcting +4.5m odometer slip to ±2cm)
    ILI-->>Pipe: 📋 Evaluate Modified ASME B31G burst pressure -> schedule excavation only on Weld #GW_18420 (Confirm)!
    Drill->>Drill: Rig #3 at 4,200m shows friction factor jump to 0.44 (>0.38 threshold) & +180 psi standpipe pressure creep
    Drill-->>Pipe: ⚡ Command immediate ROP halt, string reciprocation & hi-vis sweep 15m before pack-off (saving $1.9M NPT)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Static 5% + Odometer Excel | 🟨 Tier 2: Prompt SCADA/Mud LLM | 🟦 Tier 3: Uncompensated Flow ML | 🟩 Tier 4: `RTTM` + `DTW B31G` + `Torque-Drag` |
|---|---|---|---|---|
| **`0.8%` Pinhole Leak Detection Time** | `180.0 min` ✕ (below 5% threshold) | `95.0 min` ✕ | `28.0 min` | 🏆 **`1.8 min` (compressible `RTTM` + `NPW ToF`)** |
| **False Emergency Shut-In Rate** | `19.5%` ✕ (thermal/pump swings) | `24.0%` ✕ | `8.2%` ✕ (linepack confusion) | 🏆 **`0.3%` (`API 1130` PDE mass-momentum balance)** |
| **Unnecessary `$45K` Dig Overkill** | `38.0%` ✕ (4.5m odometer slip) | `41.5%` ✕ | `16.5%` ✕ | 🏆 **`1.4%` (Girth-weld `DTW` `±2cm` + `ASME B31G`)** |
| **Drilling Stuck-Pipe `NPT` Hours** | `64.0 hrs/mo` ✕ (reactive stall) | `58.0 hrs/mo` ✕ | `26.0 hrs/mo` | 🏆 **`3.5 hrs/mo` (15-min soft-string `Torque-Drag` lead)** |
| **Monthly Shut-In, Dig & `NPT` Loss** | `$980,000 / mo` | `$1,140,000 / mo` | `$460,000 / mo` | **`$96,000 / mo` 🏆 (-90.2%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Pipeline Midstream RTTM Leak, ILI Corrosion & Drilling Sizer](../reports/enterprise-pipeline-midstream-leak-ili-corrosion-drilling-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-pipeline-midstream-leak-ili-corrosion-drilling-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_pipeline_midstream_leak_ili_corrosion_drilling_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_pipeline_midstream_leak_ili_corrosion_drilling_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Compensate All Pipeline Flow-Balance Models with an `API 1130` Compressible Real-Time Transient Model (`RTTM`) and Couple Acoustic `NPW` Time-of-Flight**: Eliminate pump-startup and thermal-expansion false alarms while pinpointing `<1%` pinhole leaks within `±50m`.
- [ ] **Lock Multi-Run Smart-Pig (`ILI MFL / UT`) Datasets to Girth-Weld Magnetic Ring Landmarks via Dynamic Time Warping (`DTW`)**: Never match multi-year corrosion pits using raw mechanical odometer distances.
- [ ] **Gate All `$45,000` Integrity Excavations on `Modified ASME B31G / API 579` Interacting-Defect Burst Pressure**: Prioritize digs strictly where projected Safe Working Pressure falls below $1.25 \times \text{MAOP}$.
- [ ] **Track Real-Time `WITSML` Hookload and Surface Torque Residuals Against a Soft-String `Torque & Drag` Model**: Trigger automated pack-off reciprocation and sweep protocols 15 minutes before pipe sticking.
- [ ] **Clamp All Managed Pressure Drilling (`MPD`) and Mud-Weight Recommendations Inside the Deterministic `[Pore Pressure + Trip Margin, Fracture Gradient]` Window**: Prevent AI advisory systems from inducing underbalanced well kicks.
