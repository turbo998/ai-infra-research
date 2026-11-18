# Enterprise Maritime Container Shipping, Vessel Weather-Routing (`IMO CII / EU ETS`), Port Berth Congestion Prediction & Demurrage/Detention (`D&D`) Charterparty Laytime Agent Architecture Selection Guide: Great-Circle Hurry-Up-and-Wait + Manual `SOF` Excel vs. Prompt-Only Charterparty LLM vs. Open-Sea Weather Router vs. 3D Metocean Wave-Resistance Isochrone + `AIS` Port-Queue `JIT` Speed Optimizer + Deterministic `BIMCO Laytime` Interval `FSM` Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-maritime-shipping-weather-routing-port-demurrage-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, VPs of Maritime Fleet Operations & Chartering, Global Supply-Chain & Freight Forwarding (`NVOCC`) Leaders, and Maritime Legal/Laytime Claims Teams.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection/results.json`](../../experiments/enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection/results.json).

![Enterprise Maritime Weather-Routing, Port JIT & Demurrage Laytime Agent Summary](../../assets/svg/enterprise-maritime-shipping-weather-routing-port-demurrage-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating or chartering a **65-vessel maritime fleet (`850,000 TEU/year`, `280` port calls and laytime audits/month, `$720/MT` combined `VLSFO` bunker + `EU ETS` carbon price)** exposes three multimillion-dollar operational leaks:
1. **"Hurry-Up-and-Wait" Steaming Wastes Cubic Propulsion Fuel & `EU ETS` Carbon Allowances**: Because marine propulsion power scales with the **cube of vessel speed ($P \propto V^3$)**, steaming at `19.5 knots` along a fixed great-circle track (Tier 1) only to sit at anchorage for **48 hours** outside a congested port wastes **`410 MT` of VLSFO (`$295,200`)** on a single voyage and risks a **`31.0%` `IMO CII` carbon rating downgrade**.
2. **Prompt-Only LLMs Fail `BIMCO` Laytime Arithmetic (`29.0%` Demurrage Overpayment)**: Auditing a shipowner's Demurrage invoice against a 12-page Statement of Facts (`SOF`) requires exact legal state transitions—such as **`WWD SHEX UU` (Weather Working Days, Sundays & Holidays Excepted Unless Used)** prior to laytime expiry versus **"Once on Demurrage, Always on Demurrage"** after expiry. Asking a general-purpose LLM to calculate laytime in text (Tier 2) double-counts overlapping stoppages and misplaces the expiry boundary, resulting in **`29.0%` demurrage overpayments**.
3. **FIFO Container Drayage Breaches 10-Day Detention Free-Time Clocks**: Returning newly arrived containers while leaving 380 older boxes past Day 10 triggers `$165/day/container` escalating detention penalties.

Evaluated across **65 vessels, 280 monthly port calls/laytime audits, and `$720/MT` effective fuel + carbon cost**, the four architectural tiers compare as follows:

| Architectural Tier | Core Stack Combination | Bunker & `EU ETS` Overburn Rate | Port Berth `ETA MAE` | Demurrage Overpayment Rate | `IMO CII` Downgrade Risk | Monthly Bunker & Demurrage Waste | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Great-Circle Hurry-Up-and-Wait + Manual SOF** | 19.5kt full speed into port anchorage queue + manual Excel Statement-of-Facts audit | `18.2%` ✕ | `32.0 hrs` ✕ | `24.5%` ✕ | `31.0%` ✕ | `$780,000 / mo` ✕ | ⛔ Burns cubic fuel just to wait at anchor |
| **🟨 Tier 2: Prompt-Only Charterparty & Port-News LLM** | Feeding BIMCO PDFs & SOF logs into an LLM to compute laytime and suggest speeds | `16.5%` ✕ | `28.5 hrs` ✕ | `29.0%` ✕ *(breaks expiry rule)* | `27.5%` ✕ | `$820,000 / mo` ✕ | ⛔ Fails legal interval math & naval physics |
| **🟦 Tier 3: Open-Sea Weather Router Only** | Weather routing without AIS terminal crane-queue sync or BIMCO Laytime FSM | `8.4%` | `11.5 hrs` | `14.2%` ✕ | `11.0%` | `$365,000 / mo` ✕ | ⚠️ Avoids storms, still queues at port |
| **🟩 Tier 4: 3D Wave Isochrone + `AIS` Port `JIT` + `BIMCO Laytime FSM` Agent** | **3D Metocean `Kwon` Polar Isochrone + `AIS` Berth-Queue Cubic `JIT` Speed Pacing + `IMO CII` Carbon Bound + Deterministic `BIMCO Laytime` Interval `FSM`** | **`1.2%` 🏆** *(-17.0 pts)* | **`1.8 hrs` 🏆** *(JIT berth sync)* | **`0.0%` 🏆** *(Exact audit)* | **`0.0%` 🏆** *(Grade A/B)* | **`$68,000 / mo` 🏆** *(-91.3%)* | 🌟 **Recommended Architecture** |

> 💡 **Golden Rule for Maritime Routing & Supply-Chain AI Architects**: **"Synchronize open-sea 3D wave-resistance weather routing directly with destination `AIS` berth-queue predictions via cubic speed-power (`P ∝ V^3`) Virtual Arrival pacing, and audit every Demurrage & Detention (`D&D`) invoice through a deterministic `BIMCO Laytime` interval-algebra state machine rather than LLM text arithmetic."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Real-World Maritime & Laytime Pain Points)

### Q1: How does coupling 3D Metocean Wave Isochrones with `AIS` Port-Queue `JIT` Pacing save `410 MT` of fuel (`$295,200`) on a single congested voyage?
- **First-Principles Reality**:
  1. **The Admiralty Cubic Propulsion Law ($P \propto V^3$)**: Reducing vessel speed by `22%` (from `19.5 knots` to `15.2 knots`) reduces main-engine fuel consumption per hour by **`52.6%`**! When real-time `AIS` terminal berth-occupancy and crane-productivity graphs show a 48-hour queue at Rotterdam (`1.8h` ETA MAE), Tier 4 invokes the **`BIMCO Virtual Arrival` clause** to slow-steam at `15.2 knots`, arriving right as the berth opens and saving `410 MT` of VLSFO + `EU ETS` allowances (`$295,200`).
  2. **3D Wave-Resistance Isochrones & Parametric Roll Prevention**: Using hull-specific **`Kwon` speed-loss polar curves** over `ECMWF` directional wave spectra, Tier 4 routes around `7.5m` head-sea storm cells and parametric-roll resonance zones while riding favorable ocean currents (`1.2%` overburn, `0.0%` `IMO CII` downgrade risk).

### Q2: Why do prompt-only LLMs overpay Demurrage claims by `29.0%`, and how does the Deterministic `BIMCO Laytime` Interval `FSM` save `$64,200` per claim?
- **Two-Regime Legal State Machine (`On Laytime` vs. `On Demurrage`)**:
  - Under a `BIMCO` charterparty specifying `96 hours WWD SHEX UU`, rain stoppages, crane breakdowns, and Sunday non-working hours are excluded **only prior to the true laytime expiry timestamp $t_{\text{exp}}$** (with simultaneous rain + crane breakdown intervals merged via **Boolean Interval Union** so they are never double-deducted).
  - Once the accumulated working clock reaches $t_{\text{exp}}$, the state machine flips to **`ON_DEMURRAGE`** (*"Once on Demurrage, Always on Demurrage"*), where subsequent weekend and rain hours run continuously unless explicitly excepted in the demurrage clause. Tier 4's deterministic interval `FSM` eliminates `100%` of calculation errors (`0.0%` overpayment).

### Q3: How does Tier 4 eliminate escalating container Detention (`D&D`) charges across 4,200 imported containers?
- **`UIIIA` Free-Time Clock + Street-Turn Min-Cost Network Flow**:
  - Instead of dispatching drayage trucks FIFO, Tier 4 solves a daily **Min-Cost Network Flow** matching each container's carrier-specific free-time expiration clock against export factory reload locations (**Grey-Pool Street-Turns**) and off-dock empty depot gate slots.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Maritime JIT/Laytime Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Maritime Fleet Routing & Demurrage/Detention Management<br/>(65 Vessels + 850K TEU/Yr + 280 Monthly Port Calls)"] --> B{"Steams at full speed into port anchorage queues<br/>or audits SOF laytime invoices via prompt-only LLM?"}
    B -- "Yes (Hurry-and-wait or LLM laytime math)" --> C["⛔ Fuel & Demurrage Bleed (Tier 1 / Tier 2)!<br/>16.5%–18.2% bunker/ETS overburn & 24.5%–29.0% demurrage overpayment"]
    B -- "Uses open-sea weather routing" --> D{"Couples AIS berth-queue JIT speed pacing, IMO CII bounds<br/>& a deterministic BIMCO Laytime interval FSM?"}
    D -- "No (Open-sea weather router only)" --> E["⚠️ Anchorage Waiting & SOF Errors (Tier 3)<br/>11.5h berth ETA error & 14.2% demurrage overpayment"]
    D -- "Yes (3D Isochrone + Port JIT + BIMCO FSM)" --> F["🟩 Tier 4: 3D Wave Isochrone + AIS Port JIT + BIMCO Laytime FSM 🏆<br/>(1.2% Overburn | 1.8h Berth ETA MAE | 0.0% Demurrage Overpay)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 3D Metocean Isochrone + `AIS` Port `JIT` + `BIMCO Laytime FSM` Topology

```mermaid
flowchart LR
    subgraph Ocean["1. 3D Metocean & Carbon Routing Plane"]
        Met["🌊 ECMWF Wave Spectra + Currents<br/>+ Hull Kwon Speed-Loss Polars"] --> Iso["🧭 3D Dynamic Isochrone Solver<br/>Avoids Parametric Roll & Locks IMO CII A/B"]
    end

    subgraph Port["2. AIS Berth Queue & Cubic JIT Pacing"]
        AIS["⚓ Terminal AIS Draft & Crane Telemetry"] --> Queue["⏱️ Port Berth Queue Graph<br/>(1.8h ETA MAE)"]
        Iso --> JIT["⚡ Cubic P ∝ V^3 Virtual Arrival Optimizer<br/>19.5kts -> 15.2kts Saves 410 MT VLSFO"]
        Queue --> JIT
    end

    subgraph Legal["3. BIMCO Laytime FSM & Container Flow"]
        SOF["📄 Signed Statement of Facts (SOF) + CP"] --> FSM["🛡️ BIMCO Laytime Interval-Algebra FSM<br/>WWD SHEX UU Union & Once-on-Demurrage<br/>(0.0% Overpayment)"]
        Boxes["📦 4,200 Import Container Clocks"] --> Flow["🚚 UIIIA Street-Turn Min-Cost Flow"]
    end
```

### 3.3 End-to-End Sequence: Rotterdam `JIT` Virtual Arrival & `$64,200` Laytime Claim Reduction

```mermaid
sequenceDiagram
    autonumber
    participant Ship as 14,000-TEU Vessel (#VOY-NLRTM-42)
    participant JIT as 3D Isochrone + AIS Berth Queue Engine
    participant Desk as Fleet Ops & Chartering Desk
    participant FSM as BIMCO Laytime Interval FSM

    JIT->>JIT: 1,400 nm out from Rotterdam, AIS crane-queue graph predicts 48h berth congestion delay (1.8h MAE)
    JIT->>Desk: Propose BIMCO Virtual Arrival slow-steam: reduce speed 19.5kts -> 15.2kts with zero berth delay
    Desk->>Ship: ⚡ Execute 15.2kt speed profile (Act automatically) & deflect south of 7.5m head-sea storm cell
    Note over Ship,Desk: Vessel berths on arrival, saving 410 MT VLSFO + EU ETS ($295,200) and locking IMO CII Grade A!
    Desk->>FSM: Upload shipowner's $185,000 demurrage claim + 12-page Statement of Facts (SOF)
    FSM->>FSM: Compute boolean union of pre-expiry Tuesday rain (14h) + crane breakdown (4h net) + Sunday SHEX UU
    FSM-->>Desk: ✅ Output auditable laytime timesheet (Confirm): verified demurrage is $120,800, cutting $64,200 overcharge!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Great-Circle + Excel | 🟨 Tier 2: Prompt Charterparty LLM | 🟦 Tier 3: Open-Sea Router Only | 🟩 Tier 4: 3D Isochrone + Port `JIT` + `BIMCO FSM` |
|---|---|---|---|---|
| **Bunker & `EU ETS` Overburn Rate** | `18.2%` ✕ (steams fast to wait) | `16.5%` ✕ | `8.4%` (ignores berth queue) | 🏆 **`1.2%` (`P ∝ V^3` Virtual Arrival + currents)** |
| **Port Berth Arrival `ETA MAE`** | `32.0 hrs` ✕ | `28.5 hrs` ✕ | `11.5 hrs` | 🏆 **`1.8 hrs` (`AIS` draft + crane productivity graph)** |
| **Demurrage Overpayment Rate** | `24.5%` ✕ (manual Excel errors) | `29.0%` ✕ (breaks expiry logic) | `14.2%` ✕ | 🏆 **`0.0%` (deterministic `BIMCO Laytime` interval `FSM`)** |
| **`IMO CII` Rating Downgrade Risk** | `31.0%` ✕ (D/E rating penalty) | `27.5%` ✕ | `11.0%` | 🏆 **`0.0%` (voyage-level `gCO2/dwt-nm` envelope)** |
| **Monthly Bunker & `D&D` Waste** | `$780,000 / mo` | `$820,000 / mo` | `$365,000 / mo` | **`$68,000 / mo` 🏆 (-91.3%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Maritime Weather-Routing, Port JIT & Demurrage Sizer](../reports/enterprise-maritime-shipping-weather-routing-port-demurrage-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-maritime-shipping-weather-routing-port-demurrage-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Couple Open-Sea Weather Routing with Destination `AIS` Berth-Queue Predictions (`Virtual Arrival`)**: Automatically pace speed via the cubic law ($P \propto V^3$) whenever destination berth wait times exceed 6 hours.
- [ ] **Model Hull-Specific `Kwon` Wave Speed-Loss Polars and Parametric Roll Resonance Zones**: Never route laden container vessels along static geometric great circles through severe head/quartering seas.
- [ ] **Embed `IMO CII` Intensity Trajectories and `EU ETS` Carbon Allowances Directly into the Voyage Cost Function**: Prevent year-end `CII` rating downgrades.
- [ ] **Audit All Statement-of-Facts (`SOF`) Laytime Claims via a Deterministic `BIMCO` Interval-Algebra State Machine (`FSM`)**: Use LLMs only to extract structured `SOF` event timestamps; compute interval unions and *"Once on Demurrage"* transitions deterministically.
- [ ] **Optimize Inland Container Drayage Against `UIIIA` Free-Time Clocks and Grey-Pool Street-Turns**: Prioritize containers approaching detention cliff dates.
