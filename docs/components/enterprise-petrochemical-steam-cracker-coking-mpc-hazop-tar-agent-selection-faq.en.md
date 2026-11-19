# Enterprise Petrochemical & Refining Olefin Steam-Cracker (`Naphtha / Ethane Pyrolysis`), Online `NIR PIONA` Feed-Forward, Radiant-Coil Coking & Tube-Metal Temperature (`TMT <= 1085°C`) `MPC`, `IEC 61511 SIS` Safety Interlock & Turnaround (`TAR SIMOPS`) Agent Architecture Selection Guide: Fixed `COT` PID + 45-Day Decoking vs. Prompt-Only DCS Log LLM vs. Unconstrained Black-Box ML vs. Online `NIR PIONA` + Coking-Kinetic `TMT MPC` + `SIL-3 SIS` Clamp + 3D `TAR SIMOPS MILP` Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-petrochemical-steam-cracker-coking-mpc-hazop-tar-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Chief Olefin & Refinery Process Engineers, Advanced Process Control (`APC: AspenTech DMC3 / Honeywell Forge / SPYRO`) Leads, Process Safety (`IEC 61511 HAZOP / SIS`) Directors, and Turnaround (`TAR`) Planners.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection/results.json`](../../experiments/enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection/results.json).

![Enterprise Petrochemical Steam-Cracker Coking MPC, SIS & Turnaround TAR Agent Summary](../../assets/svg/enterprise-petrochemical-steam-cracker-coking-mpc-hazop-tar-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating a **1.5 million tonnes/year ethylene steam-cracking complex with `10` pyrolysis furnaces, `480` high-alloy radiant coils, cryogenic fractionation trains, and `3,500` plant turnaround (`TAR`) work orders** exposes three high-consequence failure modes:
1. **Holding Fixed Coil-Outlet-Temperature (`COT`) Setpoints Across Naphtha Feedstock Tank Switches (Tier 1) Triggers Rapid Coking and Forces `18.4` Premature Decoking Shutdowns/Year (`45-Day` Runs & `6.8%` Olefin Yield Loss)**: Switching from a light paraffinic naphtha (`72%` paraffins) to a heavy naphthenic parcel (`34%` naphthenes) without adjusting steam-to-oil dilution over-cracks naphthenes into polyaromatic coke!
2. **Unconstrained Black-Box Yield Neural Nets (Tier 3) Push Radiant-Coil Tube-Metal Temperature (`TMT`) Past the `1,085°C` `HP-40Nb` Carburization Creep Limit (`$1.2M` Tube Rupture), While Prompt-Only LLMs (Tier 2) Recommend Adding Hydrogen During Acetylene Reactor Hot-Spots (Triggering Thermal Runaway Flaring)**.
3. **Manual Turnaround (`TAR`) Scheduling Issues Conflicting Hot-Work Welding and Hydrocarbon Line-Breaking Permits in the Same 15m Zone (`SIMOPS`), Causing `21.5%` Critical-Path Schedule Slip (`$2.8M/Day` Lost Margin)**.

Evaluated across a **1.5M tonnes/yr ethylene complex, 10 furnaces, 480 radiant coils, and 3,500 TAR work orders**, the four petrochemical architectures compare as follows:

| Architectural Tier | Core Stack Combination | Premature Decoking & Trips / Yr | Olefin Yield Loss (`%`) | Tube Rupture / Runaway Risk | Turnaround (`TAR`) Slip | Monthly Yield, Decoking & TAR Loss TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Fixed `COT` PID + 45d Decoking** | Constant COT setpoint + rigid 45-day decoking + manual TAR schedules | `18.4 / yr` ✕ | `6.8%` ✕ *(feedstock shock loss)* | `3.4%` ✕ *(hot-spot blind spots)* | `21.5%` ✕ *(SIMOPS permit clashes)* | `$1,350,000 / mo` ✕ | ⛔ Short 45-day runs & costly TAR delays |
| **🟨 Tier 2: Prompt-Only DCS Alarm & Log LLM** | Feeding raw DCS alarms & shift logs into an LLM for setpoint advice | `16.2 / yr` ✕ | `7.5%` ✕ | `4.8%` ✕ *(C2H2 runaway advice)* | `19.0%` ✕ | `$1,480,000 / mo` ✕ | ⛔ Dangerous exothermic reactor hallucinations |
| **🟦 Tier 3: Unconstrained Empirical Yield ML** | Black-box ML chasing ethylene yield without coking physics or TMT clamp | `6.4 / yr` | `2.9%` | `1.1%` ✕ *(exceeds 1,085°C TMT)* | `9.2%` ✕ | `$490,000 / mo` ✕ | ⚠️ Overheats radiant tubes chasing short-term yield |
| **🟩 Tier 4: Online `NIR PIONA` + Coking `TMT MPC` + `SIL-3 SIS` + `TAR MILP` Agent** | **Online `NIR PIONA` Feed-Forward + Coking `TMT <= 1085°C MPC` + `IEC 61511 SIL-3` Clamp + 3D `TAR SIMOPS MILP`** | **`0.4 / yr` 🏆** *(78-day runs)* | **`0.5%` 🏆** *(-6.3 pts)* | **`0.0%` 🏆** *(Zero tube burst)* | **`1.4%` 🏆** *(-20.1 pts)* | **`$110,000 / mo` 🏆** *(-91.9%)* | 🌟 **Recommended Petrochemical Complex Standard** |

> 💡 **Golden Rule for Petrochemical Process Control & Safety AI Architects**: **"Never run pyrolysis furnaces on fixed COT setpoints across changing naphtha tanks or allow unconstrained ML to chase olefin yield past metallurgical limits—feed real-time `Online NIR PIONA` composition into a first-principles coking-kinetic `MPC` that balances pass-level feed and dilution steam to clamp Tube-Metal Temperature (`TMT <= 1,080°C`, extending run lengths from 45 to 78 days), enforce air-gapped `IEC 61511 SIL-3 SIS` hardware trips on exothermic hydrogenation reactors, and schedule Turnaround (`TAR`) work orders via a 3D `SIMOPS MILP` that separates hot work from hydrocarbon line-breaking by `>=15 meters`."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Petrochemical Cracking & Safety Pain Points)

### Q1: How does Online `NIR PIONA` Feed-Forward + Coking-Kinetic `TMT MPC` extend furnace run lengths from `45 days` to `78 days` with `0.0%` tube ruptures?
- **First-Principles Reality**:
  - As internal coke thickness $\delta_{\text{coke}}$ builds inside a radiant coil, the thermal resistance of the coke layer forces the outer **Tube-Metal Temperature (`TMT`)** upward for a given process gas temperature $T_{\text{gas}}$:
    $$T_{\text{TMT}, k} = T_{\text{gas}} + q_{\text{flux}} \cdot \left(\frac{\delta_{\text{coke}, k}}{\lambda_{\text{coke}}} + \frac{d_{\text{wall}}}{\lambda_{\text{metal}}}\right) \le 1,085^\circ\text{C}$$
  - Exceeding `1,085°C` causes rapid carburization and creep rupture of `HP-40Nb` alloy tubes (`$1.2M` outage).
  - Tier 4 pairs **60-second Online Near-Infrared (`NIR`) `PIONA` feedstock analyzers** with pass-level coking differential equations (cross-validated by **IR pyrometers + Venturi nozzle sonic $\Delta P$ + `TLE` inlet temperature**). When Pass #3 approaches `1,072°C`, the `MPC` trims that pass's hydrocarbon feed by `-4%` and raises dilution steam—equalizing coking across all passes and extending run lengths to **`78 days` (`0.4` trips/yr, `0.5%` yield loss)**.

### Q2: Why do Prompt-Only LLMs trigger thermal runaways on Acetylene Hydrogenation Reactors, and how does the `IEC 61511 SIL-3` Clamp prevent them?
- **Selective vs. Runaway Ethylene Hydrogenation + Air-Gapped Hardware `SIS`**:
  - In a C2 front-end/tail-end acetylene converter, a `+6°C` catalyst bed hot-spot with dropping `CO` moderator (`1.2 -> 0.3 ppm`) indicates incipient runaway hydrogenation of ethylene into ethane ($\Delta H = -136\text{ kJ/mol}$). A general LLM (Tier 2) misinterprets the temperature rise and suggests *increasing* hydrogen flow—which accelerates the exothermic runaway!
  - Tier 4's kinetic `MPC` immediately trims the $\text{H}_2/\text{C}_2\text{H}_2$ molar ratio while an **air-gapped `IEC 61511 SIL-3` Safety Instrumented System (`SIS`) hardware PLC** enforces immutable temperature-rate trip bounds (`0.0%` safety breach).

### Q3: How does the 3D `SIMOPS` Turnaround (`TAR`) `MILP` slash critical-path schedule slip from `21.5%` to `1.4%`?
- **Spatial Conflict Lockout (`Hot-Work vs. Line-Breaking`) + Heavy-Crane Sequencing**:
  - Enforces `API RP 2219` 3D spatial separation ($\text{Dist}_{3\text{D}}(\text{HotWork}_i, \text{Blinding}_j) \ge 15.0\text{m}$ and zero vertical drop overlap) inside the electronic Permit-to-Work (`e-PTW`) and 600-tonne crane schedule (`1.4%` slip).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + PIONA/TMT-MPC/TAR Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Petrochemical Olefin Steam-Cracker, SIS & Turnaround<br/>(1.5M t/yr Ethylene + 10 Furnaces + 480 Coils + 3,500 TAR Work Orders)"] --> B{"Uses fixed COT PID setpoints + 45-day decoking<br/>or prompt-only LLM DCS alarm advice?"}
    B -- "Yes (Fixed COT PID or prompt-only LLM)" --> C["⛔ Frequent Decoking Trips, Yield Loss & Runaway Risk (Tier 1 / Tier 2)!<br/>18.4 trips/yr, 6.8% olefin yield loss, 4.8% tube/runaway risk & 21.5% TAR slip"]
    B -- "Uses advanced process control / ML" --> D{"Combines online NIR PIONA feed-forward, coking-kinetic<br/>TMT (<=1085°C) MPC, SIL-3 SIS clamps & 3D SIMOPS TAR MILP?"}
    D -- "No (Unconstrained black-box yield neural net)" --> E["⚠️ Radiant Coil Overheating & Carburization Risk (Tier 3)<br/>Chases short-term yield past 1,085°C HP-40Nb limit (1.1% tube rupture risk)"]
    D -- "Yes (PIONA + Coking TMT MPC + SIL-3 + SIMOPS MILP)" --> F["🟩 Tier 4: NIR PIONA + Coking TMT MPC + SIL-3 SIS + TAR MILP Agent 🏆<br/>(78-Day Runs | 0.5% Yield Loss | 0.0% Tube Rupture | 1.4% TAR Slip)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Online `NIR PIONA` + Coking-Kinetic `TMT MPC` + `SIL-3 SIS` + `TAR SIMOPS` Topology

```mermaid
flowchart LR
    subgraph Feed["1. Online NIR PIONA Feed-Forward Plane"]
        NIR["🔬 Header Online NIR Spectrometer<br/>60s Paraffin/Naphthene/Aromatic"] --> MPC["⚙️ SPYRO + Coking-Kinetic Hybrid MPC<br/>Modulates Steam/Oil Ratio & Pass Bias<br/>(0.5% Olefin Yield Loss)"]
    end

    subgraph Furnace["2. Coil TMT (<=1085°C) & SIL-3 SIS Gate"]
        MPC --> Coil["🔥 IR Pyrometer + Venturi ΔP + TLE ΔT<br/>Clamps HP-40Nb Skin <=1,080°C<br/>(Extends Run from 45d to 78d)"]
        Coil --> SIS["🛡️ Air-Gapped IEC 61511 SIL-3 PLC<br/>C2H2 Hydrogenation Runaway Clamp (0.0% Breach)"]
    end

    subgraph TAR["3. Turnaround 3D SIMOPS & Crane MILP"]
        WO["🗓️ 3,500 TAR Work Orders + Blind List"] --> SIMOPS["🚧 3D Spatial SIMOPS & Crane MILP<br/>Blocks Hot-Work <15m from Line-Breaking<br/>(1.4% Critical-Path Slip)"]
    end
```

### 3.3 End-to-End Sequence: Handling a `34%` Naphthenic Feed Switch & Preventing a `1,072°C` Tube Rupture

```mermaid
sequenceDiagram
    autonumber
    participant Tank as Naphtha Tank Farm & Online NIR
    participant MPC as Coking-Kinetic & PIONA Hybrid MPC
    participant SIS as Furnace #4 & C2H2 Converter SIL-3 SIS
    participant TAR as Turnaround 3D SIMOPS MILP

    Tank->>MPC: Tank switch to heavy parcel: Online NIR detects naphthenes jumping from 18% to 34% (15 mins upstream)
    MPC->>SIS: ⚡ Feed-forward raise dilution steam +0.08 kg/kg & adjust COT -> hold P/E severity at 0.54 without coking surge!
    SIS-->>MPC: Day 52: Furnace #4 Pass #3 IR pyrometer + Venturi ΔP confirm 4.8mm coke layer & TMT = 1,072°C (near 1,085°C limit)
    MPC->>SIS: Trim Pass #3 hydrocarbon feed -4% & boost pass steam -> cool TMT to 1,064°C and extend run length to 78 days!
    TAR->>TAR: Evaluate TAR Work Order #WO-1142 (Furnace #2 welding): detect active Quench Tower flange blinding 8.2m away (<15m limit)
    TAR-->>SIS: 🚨 Hold hot-work e-PTW permit until Quench Tower blind flange torque & N2 purge test pass (1.4% TAR slip)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Fixed `COT` + 45d Run | 🟨 Tier 2: Prompt DCS Log LLM | 🟦 Tier 3: Unconstrained Yield ML | 🟩 Tier 4: `PIONA` + Coking `MPC` + `SIL-3` |
|---|---|---|---|---|
| **Premature Decoking & Trips / Yr** | `18.4 / yr` ✕ (45-day rigid cycle) | `16.2 / yr` ✕ | `6.4 / yr` | 🏆 **`0.4 / yr` (`78-day` extended runs, `-97.8%`)** |
| **Olefin Yield Loss (`%`)** | `6.8%` ✕ (uncompensated feed shifts) | `7.5%` ✕ | `2.9%` | 🏆 **`0.5%` (60s `Online NIR PIONA` feed-forward `MPC`)** |
| **Tube Rupture / Runaway Risk** | `3.4%` ✕ (manual peephole walks) | `4.8%` ✕ (bad H2 runaway advice) | `1.1%` ✕ (pushes `TMT > 1,085°C`) | 🏆 **`0.0%` (pass-level `TMT <= 1,080°C` + `SIL-3` clamp)** |
| **Turnaround (`TAR`) Schedule Slip** | `21.5%` ✕ (SIMOPS safety stand-downs) | `19.0%` ✕ | `9.2%` ✕ | 🏆 **`1.4%` (3D `SIMOPS` 15m lockout + crane `MILP`)** |
| **Monthly Yield, Decoking & TAR Loss** | `$1,350,000 / mo` | `$1,480,000 / mo` | `$490,000 / mo` | **`$110,000 / mo` 🏆 (-91.9%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Petrochemical Steam-Cracker Coking MPC & Turnaround Sizer](../reports/enterprise-petrochemical-steam-cracker-coking-mpc-hazop-tar-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-petrochemical-steam-cracker-coking-mpc-hazop-tar-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Install Online Near-Infrared (`NIR`) `PIONA` Analyzers on Naphtha Feed Headers Linked to the Cracking `MPC`**: Compensate steam-to-hydrocarbon dilution ratios and `COT` proactively before tank-switch parcels hit the radiant coils.
- [ ] **Cross-Validate Optical `TMT` Pyrometers Against Venturi Nozzle Sonic Pressure-Drop ($\Delta P$) and `TLE` Inlet Temperatures**: Prevent soot-obscured peepholes from hiding localized `>1,085°C` `HP-40Nb` tube overheating.
- [ ] **Enforce Air-Gapped `IEC 61511 SIL-3` Safety Instrumented System (`SIS`) Hardware Clamps Below All AI/APC Setpoints**: Never allow an AI agent to override thermal-runaway trips on acetylene hydrogenation or pyrolysis units.
- [ ] **Embed 3D Spatial `SIMOPS` Lockout Rules (`>=15m` Separation Between Hot Work and Hydrocarbon Line-Breaking) Inside Turnaround (`TAR`) `e-PTW` and `MILP` Engines**: Eliminate safety stand-downs and crane bottlenecks.
- [ ] **Control Steam-Air Decoking via Online Effluent `CO / CO2` Infrared Analyzers**: Modulate air-to-steam ratios dynamically to prevent exothermic coke-burn overheating of coil walls.
