# Enterprise Commercial Aviation MRO (`Maintenance, Repair & Overhaul`), Jet Engine `EGT` Margin Prognostics (`RUL`), `ATA Spec 2000` Rotable Spare-Parts Pooling & `FAA Part 145` Airworthiness Sign-Off Agent Architecture Selection Guide: Fixed Cycle Hard-Time + Reactive AOG vs. Prompt-Only `PIREP/IPC` LLM vs. Uncorrected Raw Sensor `RUL` Neural Net vs. `ISA`-Corrected Gas-Path Twin + `Sherbrooke METRIC` Rotable Solver + `MSN/SB IPC Effectivity AST` + `FAA Form 8130-3` Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-aviation-mro-jet-engine-rul-rotable-spares-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, VPs of Airline Technical Operations & `Part 145` MROs, Engine Fleet Management Leads, and Aviation MRO ERP (`AMOS / TRAX / Maintenix`) Engineering Teams.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_aviation_mro_jet_engine_rul_rotable_spares_agent_selection/results.json`](../../experiments/enterprise_aviation_mro_jet_engine_rul_rotable_spares_agent_selection/results.json).

![Enterprise Aviation MRO, Jet Engine EGT Margin RUL & Rotable Spares Agent Summary](../../assets/svg/enterprise-aviation-mro-jet-engine-rul-rotable-spares-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Managing technical operations across **280 commercial aircraft, 620 turbofan engines (`LEAP / GEnx / Trent`), and a `$240M` rotable Line-Replaceable Unit (`LRU`) spare-parts pool**—where every unscheduled **Aircraft-on-Ground (`AOG`)** event costs **`$150,000/day`** in flight cancellations and passenger re-accommodation—exposes three high-cost engineering failure modes:
1. **Uncorrected Raw-Sensor `RUL` Neural Nets Confuse Hot-and-High Takeoffs with Turbine Blade Failure (`12.4%–21.0%` Wasted Engine "Green-Time")**: When an aircraft takes off from Denver (`5,430 ft` elevation, `36°C` ambient temperature, air-conditioning packs ON), raw takeoff Exhaust Gas Temperature (`EGT`) spikes to `935°C`. An uncorrected time-series neural net (Tier 3) flags terminal engine degradation and pulls a healthy engine **850 cycles early (`$420,000` wasted overhaul green-time)**!
2. **Naive `IPC` Vector RAG Ignores Aircraft Tail `MSN` & Service-Bulletin (`SB`) Effectivity Blocks (`17.8%` Wrong-Part Dispatch)**: When a line mechanic requests a High-Pressure Bleed Valve (`ATA 36-11-01`), the Illustrated Parts Catalog (`IPC`) lists four part numbers contingent on Manufacturer Serial Number (`MSN`) and `SB-36-1094` modification status. Prompt-only LLMs (Tier 2) ship the pre-modification part to the tarmac, causing a 14-hour international flight cancellation when the flange bolts do not fit!
3. **Life-Limited Part (`LLP`) Back-to-Birth Traceability Gaps Risk `FAA Part 145` Grounding**: Failing to verify unbroken cycle-since-new (`CSN`) history and `FAA Form 8130-3 / EASA Form 1` airworthiness tags on a 20,000-cycle HPT disk exposes the carrier to million-dollar civil penalties.

Evaluated across **280 aircraft tails, 620 turbofan engines, and a `$240M` rotable spares pool**, the four architectural tiers compare as follows:

| Architectural Tier | Core Stack Combination | Unscheduled `AOG` Events / Mo | `IPC` Effectivity Wrong-Part Rate | Wasted Engine `Green-Time` Cycles | `FAA 8130-3` Audit Escape | Monthly `AOG` Charter & Spares Loss | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Hard-Time Cycles + Reactive AOG** | Rigid cycle overhaul limits + reactive AOG charter couriers + manual PDF IPC lookup | `18.5 / mo` ✕ | `9.4%` ✕ | `21.0%` ✕ *(pulls healthy engines)* | `2.8%` ✕ | `$1,280,000 / mo` ✕ | ⛔ Frequent AOGs & 21% wasted engine life |
| **🟨 Tier 2: Prompt-Only `PIREP` & `IPC` LLM** | Feeding pilot squawks into an LLM to select IPC part numbers without MSN effectivity gates | `16.0 / mo` ✕ | `17.8%` ✕ *(wrong P/N on tarmac)* | `18.5%` ✕ | `6.5%` ✕ | `$1,410,000 / mo` ✕ | ⛔ Ships incompatible pre-SB parts |
| **🟦 Tier 3: Uncorrected Raw Sensor `RUL` Neural Net** | LSTM/Transformer on raw EGT without ISA sea-level correction or METRIC rotable pooling | `7.2 / mo` | `4.8%` ✕ | `12.4%` ✕ *(Denver hot-day alarm)* | `1.2%` ✕ | `$590,000 / mo` ✕ | ⚠️ Mistakes hot/high takeoff for HPT rub |
| **🟩 Tier 4: `ISA` Gas-Path Twin + `METRIC` Pooling + `MSN AST` + `FAA 8130-3` Agent** | **`ISA`-Corrected Gas-Path Twin (`ΔEGT/ΔN2/ΔFF`) + `Sherbrooke METRIC` Pooling + `MSN/SB IPC Effectivity AST` + `FAA 8130-3 LLP` Gate** | **`0.6 / mo` 🏆** *(-96.8%)* | **`0.0%` 🏆** *(Exact tail match)* | **`1.9%` 🏆** *(-19.1 pts)* | **`0.0%` 🏆** *(100% airworthy)* | **`$142,000 / mo` 🏆** *(-88.9%)* | 🌟 **Recommended Architecture** |

> 💡 **Golden Rule for Aviation MRO & Engine Prognostics Architects**: **"Never train engine `RUL` models on raw takeoff temperatures and never retrieve aircraft spare part numbers via unconstrained vector RAG—normalize gas-path telemetry (`EGT / N1 / N2 / Fuel Flow`) to `ISA` Sea-Level Static reference conditions while separating reversible compressor fouling (water-washable) from permanent HPT blade rub, position rotable LRUs across outstations via `Sherbrooke METRIC`, and gate every part dispatch on a deterministic `Tail MSN + Service-Bulletin Effectivity AST` and `FAA Form 8130-3` back-to-birth cycle ledger."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Real-World Aviation MRO Pain Points)

### Q1: Why do raw-sensor neural nets pull healthy engines after a Denver summer takeoff, and how does the `ISA` Gas-Path Thermodynamic Twin save `$420K` per engine?
- **First-Principles Reality**:
  1. **`ISA` Sea-Level Static (`SLS`) Dimensionless Normalization**: Raw takeoff `EGT` (`935°C` at Denver, `5,430 ft`, `36°C` OAT) is heavily skewed by inlet total temperature ratio $\theta = T_{\text{total}} / 288.15\text{K}$, pressure ratio $\delta = P_{\text{total}} / 101.325\text{kPa}$, and cabin pack bleed extraction. Tier 4 normalizes every snapshot to `ISA SLS` conditions:
     $$\text{EGT}_{\text{SLS}} = \frac{\text{EGT}_{\text{raw}}}{\theta^{0.86}} - \Delta\text{EGT}_{\text{bleed}}$$
     proving the Denver engine still holds a healthy **`+28°C` EGT Margin** (`1.9%` wasted green-time vs. `12.4%–21.0%`).
  2. **Separating Reversible Compressor Fouling from Permanent HPT Tip Rub**: By tracking the joint Kalman residual $(\Delta\text{EGT}, \Delta N_2, \Delta W_f)$, Tier 4 distinguishes dust-induced High-Pressure Compressor (`HPC`) fouling—which is cured in 45 minutes via an **On-Wing Water Wash**—from irreversible High-Pressure Turbine (`HPT`) blade-tip clearance erosion.

### Q2: How does the `Tail MSN + Service-Bulletin Effectivity AST` eliminate the `17.8%` wrong-part dispatch rate of LLM RAG?
- **Aircraft-Specific Configuration Intersection**:
  - In the Illustrated Parts Catalog (`IPC`), a single figure (`ATA 36-11-01`) lists multiple part numbers gated by **Effectivity Blocks** (e.g., `P/N 6984-01A` for `MSN 001–145 Pre-SB-36-1094` vs. `P/N 6984-04C` for `Post-SB-36-1094`).
  - Tier 4 intersects the `IPC` effectivity AST against the specific aircraft's live configuration tree in `AMOS / TRAX` (`Tail N408AA -> MSN 6412 -> Mod Status Post-SB-36-1094`), deterministically locking `P/N 6984-04C` (`0.0%` wrong-part dispatch).

### Q3: How do `Sherbrooke METRIC` Multi-Echelon Pooling and the `FAA Form 8130-3` Gate protect fleet reliability and airworthiness?
- **Poisson Recoverable Item Optimization + Back-to-Birth `LLP` Verification**:
  1. **`Sherbrooke METRIC` Rotable Pooling**: Couples repair-shop turnaround times (`TAT`) with fleet-wide `RUL` hazard rates to allocate `$95K` Integrated Drive Generators (`IDGs`) across 3 overhaul bases and 45 line outstations—cutting unscheduled `AOGs` from `18.5/mo` to **`0.6/mo`**.
  2. **`FAA Form 8130-3 / EASA Form 1` & `LLP` Back-to-Birth Gate**: Verifies cryptographic signatures on authorized release certificates and checks unbroken cycle-since-new (`14,210 CSN <= 20,000` hard cap) before any Life-Limited Part can be signed off onto an engine (`0.0%` audit escape).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Gas-Path/METRIC/Airworthiness Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Commercial Aviation MRO & Engine Prognostics<br/>(280 Aircraft + 620 Turbofan Engines + $240M Rotable Pool)"] --> B{"Uses rigid hard-time cycle limits<br/>or prompt-only LLM IPC part lookup?"}
    B -- "Yes (Hard-time cycles or raw IPC RAG)" --> C["⛔ High AOGs & Wrong Parts on Tarmac (Tier 1 / Tier 2)!<br/>16.0–18.5 AOGs/mo, 17.8% wrong IPC part shipped & 21% wasted engine life"]
    B -- "Uses sensor RUL models" --> D{"Combines ISA SLS gas-path normalization, Sherbrooke METRIC<br/>rotable pooling, MSN/SB Effectivity AST & FAA 8130-3 gate?"}
    D -- "No (Uncorrected raw EGT neural net)" --> E["⚠️ Hot-and-High False Engine Pulls (Tier 3)<br/>12.4% wasted green-time pulling healthy engines after hot-day takeoffs"]
    D -- "Yes (ISA Twin + METRIC + MSN AST + 8130-3)" --> F["🟩 Tier 4: ISA Gas-Path Twin + Sherbrooke METRIC + MSN Effectivity AST + 8130-3 Gate 🏆<br/>(0.6 AOG/Mo | 0.0% Wrong Part | 1.9% Wasted Green-Time | 0.0% Audit Escape)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `ISA` Gas-Path Twin + `Sherbrooke METRIC` + `MSN/8130-3` Airworthiness Topology

```mermaid
flowchart LR
    subgraph Engine["1. ISA Gas-Path Thermodynamic Twin"]
        ACARS["✈️ ACARS/QAR Takeoff & Cruise<br/>(Raw EGT / N1 / N2 / FF / Alt / OAT)"] --> ISA["🌡️ ISA Sea-Level Static Corrector<br/>Strips Altitude/Bleed Distortion<br/>Separates Water-Wash vs. HPT Rub"]
    end

    subgraph Spares["2. Sherbrooke METRIC Rotable Pool"]
        ISA -->|"Fleet RUL Hazard λ(t)"| METRIC["📦 Multi-Echelon METRIC Solver<br/>3 Overhaul Hubs + 45 Outstations<br/>(0.6 Unscheduled AOGs / Mo)"]
    end

    subgraph Airworth["3. Tail MSN Effectivity & FAA 8130-3 Gate"]
        IPC["📘 AMM / IPC + Tail MSN/SB Ledger"] --> Eff["🔒 IPC Effectivity Range AST Gate<br/>Locks Post-SB Part Number (0.0% Error)"]
        LLP["📑 Life-Limited Part (LLP) & 8130-3"] --> Sign["🛡️ Back-to-Birth Cycle & AD Gate<br/>Verifies CSN <= 20,000 Cap"]
        METRIC --> Eff
        Eff --> Sign
    end
```

### 3.3 End-to-End Sequence: Filtering a Denver Hot-Day Takeoff, Locking `MSN 6412` Part Effectivity & Verifying an `LLP` Disk

```mermaid
sequenceDiagram
    autonumber
    participant Jet as Aircraft (ACARS / PIREP Stream)
    participant Twin as ISA Gas-Path Thermodynamic Twin
    participant IPC as Tail MSN/SB AST & METRIC Pool
    participant Gate as FAA Form 8130-3 & LLP Trace Gate
    participant MCC as Maintenance Control Center (Part 145)

    Jet->>Twin: Case A: Denver takeoff (5,430 ft, 36°C) logs 935°C raw EGT; Engine ESN-912 logs +14°C ΔEGT_SLS & +1.8% ΔFF
    Twin->>Twin: Normalize via ISA theta/delta: Denver engine is healthy (+28°C margin, saving $420K); flag ESN-912 HPT rub
    Jet->>IPC: Case B: Tail N408AA (MSN 6412) at LHR requires ATA 36-11-01 High-Pressure Bleed Valve replacement
    IPC->>IPC: Intersect IPC effectivity with MSN 6412 mod ledger (Post-SB-36-1094) -> block P/N 6984-01A, lock P/N 6984-04C
    IPC->>Gate: Pull pre-positioned P/N 6984-04C from LHR METRIC pool & inspect overhauled HPT Stage-1 Disk
    Gate->>Gate: Verify digital FAA Form 8130-3 signature & unbroken back-to-birth ledger (14,210 CSN <= 20,000 limit)
    Gate-->>MCC: ✅ Issue compliant Part 145 Airworthiness Release (Act automatically / Confirm) with 0.0% wrong-part delay!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Hard-Time + Reactive AOG | 🟨 Tier 2: Prompt `PIREP/IPC` LLM | 🟦 Tier 3: Raw Sensor `RUL` NN | 🟩 Tier 4: `ISA` Twin + `METRIC` + `MSN AST` |
|---|---|---|---|---|
| **Unscheduled `AOG` Events / Month** | `18.5 / mo` ✕ (outstation stockouts) | `16.0 / mo` ✕ | `7.2 / mo` | 🏆 **`0.6 / mo` (30-cycle lead + `METRIC` pooling)** |
| **`IPC` Effectivity Wrong-Part Rate** | `9.4%` ✕ (manual SB footnote slips) | `17.8%` ✕ (ignores MSN/SB blocks) | `4.8%` ✕ | 🏆 **`0.0%` (`Tail MSN + SB` ledger $\cap$ `IPC AST`)** |
| **Wasted Engine `Green-Time` Cycles** | `21.0%` ✕ (rigid cycle pulls) | `18.5%` ✕ | `12.4%` ✕ (hot-day false alarms) | 🏆 **`1.9%` (`ISA SLS` correction + water-wash split)** |
| **`FAA 8130-3 / LLP` Audit Escape** | `2.8%` ✕ (paper trace gaps) | `6.5%` ✕ | `1.2%` ✕ | 🏆 **`0.0%` (`8130-3` signature + back-to-birth `CSN`)** |
| **Monthly `AOG` & Spares TCO** | `$1,280,000 / mo` | `$1,410,000 / mo` | `$590,000 / mo` | **`$142,000 / mo` 🏆 (-88.9%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Aviation MRO, Jet Engine RUL & Rotable Spares Sizer](../reports/enterprise-aviation-mro-jet-engine-rul-rotable-spares-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-aviation-mro-jet-engine-rul-rotable-spares-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_aviation_mro_jet_engine_rul_rotable_spares_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_aviation_mro_jet_engine_rul_rotable_spares_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Normalize All Engine Gas-Path Telemetry (`EGT / N1 / N2 / Fuel Flow`) to `ISA` Sea-Level Static (`SLS`) Conditions Before `RUL` Scoring**: Strip out airport elevation, ambient temperature, Mach number, and cabin bleed-air effects.
- [ ] **Decouple Reversible Compressor Dust Fouling from Irreversible HPT Blade-Tip Rub**: Trigger a 45-minute on-wing water wash when $(\Delta N_2 / \Delta W_f)$ indicates compressor fouling instead of scheduling an engine shop visit.
- [ ] **Gate Every `IPC` Spare-Part Dispatch on a Deterministic `Aircraft MSN + Service-Bulletin Effectivity AST`**: Never dispatch an LRU without verifying post-SB modification compatibility for that exact tail number.
- [ ] **Optimize Multi-Echelon Rotable LRU Allocation via `Sherbrooke METRIC` Coupled with Live Fleet `RUL` Hazard Rates**: Balance central overhaul shop buffers against line-outstation AOG protection.
- [ ] **Enforce Digital `FAA Form 8130-3 / EASA Form 1` Signature Verification and Unbroken Back-to-Birth `CSN` Ledgers for All Life-Limited Parts (`LLPs`)**: Block installation immediately if any historical operator cycle record is missing.
