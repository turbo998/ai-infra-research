# Enterprise Railway & High-Speed Rolling-Stock Predictive Maintenance, Wheel-Profile Wayside Acoustic/Laser Hot-Box Detector (`WILD / HBD`), Overhead Catenary Pantograph Arcing `VLM` & Dynamic Track-Possession Maintenance Window `MILP` Agent Architecture Selection Guide: Fixed-Mileage Wheel Lathe + Single-Threshold Alarms vs. Prompt-Only Wayside Log LLM vs. Per-Site Sensor ML vs. `RFID` Axle-Kalman `WILD/HBD/TADS` + 300 km/h Roof Catenary `VLM` + `EN 13848` Track-Possession `MILP` Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-railway-rolling-stock-catenary-track-maintenance-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Chief Mechanical & Track Engineers at High-Speed Rail, Heavy-Haul Freight & Metro Operators, Wayside Detector Network (`WILD / HBD / TADS`) Leads, and Rail Operations Control Centers.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_railway_rolling_stock_catenary_track_maintenance_agent_selection/results.json`](../../experiments/enterprise_railway_rolling_stock_catenary_track_maintenance_agent_selection/results.json).

![Enterprise Railway Rolling-Stock WILD/HBD, Catenary VLM & Track-Possession Agent Summary](../../assets/svg/enterprise-railway-rolling-stock-catenary-track-maintenance-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating a **4,200 km high-speed and heavy-haul rail network with 48,000 rolling-stock wheelsets and 160 wayside detector sites (`WILD / HBD / TADS`)**—where every unscheduled mainline service disruption costs **`$65,000/hour`** in cascading timetable delays—exposes three high-cost engineering failure modes:
1. **Single-Threshold Infrared Hot-Box Detectors (`HBD`) Confuse Sticking Brake Pads with Bearing Seizures (`24.5%` False Mainline Train-Stops)**: When a brake caliper drags slightly at `220 km/h` (or low-angle sun glint hits the pyrometer), radiated heat warms the axle journal box by `+62°C`. A single-threshold `HBD` alarm (Tier 1) halts the train on the mainline, delaying 14 following high-speed trains for a harmless brake rub!
2. **Waiting for `400 kN` Wheel Impact (`WILD`) Thresholds Destroys `21.8%` of Wheelset Rim Life**: Allowing subsurface rolling-contact fatigue (`RCF`) cracks to grow until impact forces hit `400 kN` causes deep tread shelling—requiring an **`8.0 mm` deep underfloor lathe cut** that scraps `80,000 km` of usable wheel rim steel instead of a proactive **`1.2 mm` skim cut** at `310 kN`!
3. **Unmonitored 25kV Catenary Wire Stagger (`>±300mm`) Causes `$1.4M` Pantograph Dewirings, While Manual 4-Hour Night Possessions Leave `45%` of Track Time Idle**.

Evaluated across a **4,200 km rail network, 48,000 wheelsets, and 160 wayside detector sites**, the four architectural tiers compare as follows:

| Architectural Tier | Core Stack Combination | Mainline Service Disruptions / Mo | False Emergency Train-Stop Rate | Wheelset Rim Lathe Waste | Catenary / Rail Defect Escape | Monthly Delay, Wheel & Possession Loss | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Fixed-Mileage Lathe + Static HBD/WILD** | 250,000-km fixed wheel turning + single-site static 400kN WILD / +60°C HBD train stops | `19.2 / mo` ✕ | `24.5%` ✕ *(brake-drag stops)* | `21.8%` ✕ *(8.0mm deep cuts)* | `3.4%` ✕ | `$960,000 / mo` ✕ | ⛔ Halts trains on brake rub & wastes wheel rims |
| **🟨 Tier 2: Prompt-Only Wayside & Track Log LLM** | Feeding raw wayside alert strings & inspector notes into an LLM to schedule work orders | `17.5 / mo` ✕ | `27.0%` ✕ | `19.4%` ✕ | `4.8%` ✕ *(>32°C tamping risk)* | `$1,080,000 / mo` ✕ | ⛔ Schedules ballast tamping during buckling heat |
| **🟦 Tier 3: Per-Site Wayside Anomaly ML Only** | Isolated single-site ML without RFID axle genealogy, TADS acoustic fusion, or possession MILP | `6.8 / mo` | `8.4%` ✕ | `10.5%` | `1.1%` ✕ | `$410,000 / mo` ✕ | ⚠️ Cannot separate brake heat from bearing spall |
| **🟩 Tier 4: `RFID` Axle-Kalman + Catenary `VLM` + `EN 13848` Possession `MILP` Agent** | **Multi-Site `RFID` Axle-Kalman (`HBD + TADS + WILD`) + 300 km/h Roof Catenary `VLM` + `1.2mm` Early Lathe + 4h Night Possession `MILP`** | **`0.4 / mo` 🏆** *(-97.9%)* | **`0.3%` 🏆** *(-24.2 pts)* | **`1.6%` 🏆** *(+80K km rim life)* | **`0.0%` 🏆** *(Zero dewire)* | **`$84,000 / mo` 🏆** *(-91.3%)* | 🌟 **Recommended Architecture** |

> 💡 **Golden Rule for Railway Rolling-Stock & Infrastructure AI Architects**: **"Never order an emergency mainline train stop from a single-site infrared `HBD` spike alone—link every wheelset across consecutive wayside sites via `AEI/RFID` tags to fuse journal IR, brake-disc IR, and `TADS` acoustic bearing spall signatures (`0.3%` false stops), trigger `1.2mm` minimum-cut wheel lathe turning on speed-normalized `WILD` trends before deep shelling occurs, and schedule nightly 4-hour track possessions via a multi-gang `MILP` interlocked with catenary power isolation and `32°C` rail neutral-temperature buckling bounds."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Real-World Rail Maintenance Pain Points)

### Q1: How does fusing Dual-Beam Infrared (`HBD`) with Acoustic Bearing Signatures (`TADS`) across `RFID`-linked sites cut false mainline train stops from `24.5%` to `0.3%`?
- **First-Principles Reality**:
  - A sticking brake caliper radiates heat into the adjacent axle journal housing (`+62°C`), fooling a single-beam `HBD` pyrometer (Tier 1) into halting the train on the mainline.
  - Tier 4 tracks each axle by its **`AEI / RFID` tag** across consecutive wayside sites, comparing the **Journal-to-Car-Median temperature z-score** against **Brake-Disc IR temperature (`195°C`)** and upstream **Trackside Acoustic Detection System (`TADS`)** high-frequency roller-spall decibels (`0.0 dB`). Confirming quiet acoustic bearing raceways with hot brake discs isolates a sticking brake pad—commanding a cab brake cycling reset and routing to the terminal (`0.3%` false stops).

### Q2: How does Speed-Normalized `WILD` Tracking + Proactive `1.2mm` Lathe Turning save `80,000 km` of wheelset rim life?
- **Preventing Subsurface Rolling-Contact Fatigue (`RCF`) Shelling**:
  - Raw `WILD` peak impact force scales nonlinearly with train speed $V_{\text{train}}^{1.42}$ and car loading. Tier 4 normalizes every impact to reference speed and axle load:
    $$F_{\text{norm}} = F_{\text{static}} + (F_{\text{peak}} - F_{\text{static}}) \cdot \left(\frac{V_{\text{ref}}}{V_{\text{train}}}\right)^{1.42}$$
  - Scheduling an underfloor lathe turn at `310–320 kN` normalized impact requires only a **`1.2 mm` skim cut** to remove incipient surface cracks—whereas waiting for the legacy `400 kN` alarm lets cracks propagate into deep spalls requiring an **`8.0 mm` cut** (`1.6%` rim waste vs. `21.8%`).

### Q3: How do Roof-Mounted `300 km/h` Catenary `VLMs` and Night Possession `MILPs` prevent pantograph dewirings and summer rail buckling?
- **Revenue-Train Overhead Inspection + Interlocked 4-Hour Possession Scheduling**:
  1. **300 km/h UV Arc + 3D Laser Catenary `VLM`**: Roof cameras on revenue trains measure contact-wire residual thickness (`7.9mm < 8.2mm` limit) and lateral wind stagger (`+340mm > ±300mm` limit) at full line speed (`0.0%` dewiring escape).
  2. **Night Possession `MILP` + `32°C` Buckling Lockout**: Schedules ballast tampers, rail grinders, ultrasonic cars, and catenary tower wagons across the `00:30–04:30` window under overhead power-isolation constraints while locking out ballast disturbance whenever daytime rail temperature exceeds `32°C`.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Wayside/Catenary/Possession Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Railway Rolling-Stock, Catenary & Track Maintenance<br/>(4,200 km Track + 48,000 Wheelsets + 160 Wayside Detector Sites)"] --> B{"Uses fixed-mileage wheel turning<br/>and single-site static HBD/WILD train stops?"}
    B -- "Yes (Fixed lathe + static HBD stops)" --> C["⛔ Mainline Delays & Rim Waste (Tier 1 / Tier 2)!<br/>19.2 disruptions/mo, 24.5% false stops on brake drag & 21.8% wheel rim waste"]
    B -- "Uses wayside sensor ML" --> D{"Combines RFID multi-site Axle-Kalman (HBD+TADS+WILD),<br/>300 km/h roof catenary VLM & interlocked night possession MILP?"}
    D -- "No (Isolated per-site wayside ML)" --> E["⚠️ Brake-Heat False Alarms & Window Waste (Tier 3)<br/>8.4% false train stops & 45% idle time in 4-hour night possessions"]
    D -- "Yes (Axle-Kalman + Catenary VLM + Possession MILP)" --> F["🟩 Tier 4: RFID Axle-Kalman + 300 km/h Catenary VLM + Night Possession MILP 🏆<br/>(0.4 Disruptions/Mo | 0.3% False Stops | 1.6% Rim Waste | 0.0% Defect Escape)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `RFID` Axle-Kalman + 300 km/h Catenary `VLM` + Night Possession `MILP` Topology

```mermaid
flowchart LR
    subgraph Wayside["1. Multi-Site RFID Axle-Kalman Plane"]
        Sites["160 Wayside Detector Sites<br/>(Journal IR + Disc IR + TADS + WILD)"] --> Kalman["🚆 AEI/RFID Longitudinal Kalman Filter<br/>Separates Brake Drag vs. Bearing Spall (0.3% Stop)<br/>Speed-Normalized 1.2mm Early Lathe Cut"]
    end

    subgraph Catenary["2. 300 km/h Roof Catenary VLM Plane"]
        Roof["⚡ Revenue-Train Roof 3D Laser + UV Camera"] --> VLM["👁️ Contact-Wire Wear & Stagger VLM<br/>Flags Stagger >300mm & Thickness <8.2mm<br/>(0.0% Dewiring Escape)"]
    end

    subgraph Window["3. 00:30–04:30 Night Possession MILP"]
        Kalman --> MILP["🗓️ EN 13848 Track & Multi-Gang MILP<br/>Tamping / Grinding / Ultrasonic / Catenary<br/>+ Power Interlock & >32°C Buckling Gate"]
        VLM --> MILP
    end
```

### 3.3 End-to-End Sequence: Filtering a `220 km/h` Brake-Drag False Alarm & Scheduling a Night Possession

```mermaid
sequenceDiagram
    autonumber
    participant Train as High-Speed Train (#G804, 220–300 km/h)
    participant HBD as RFID Axle-Kalman (HBD + TADS + WILD)
    participant VLM as Roof 3D Laser & UV Catenary VLM
    participant MILP as Depot Lathe & Night Possession MILP

    Train->>HBD: Pass #HBD-44: Car #18 Axle #3 logs +62°C journal rise; #WILD-08 logs 310 kN impact on Axle #0942
    HBD->>HBD: Fuse TADS acoustic bearing signature (0.0 dB spall) + Brake-Disc IR (195°C) -> confirm sticking brake pad only
    HBD->>Train: ⚡ Command brake-release cycle & continue at line speed to terminal (Act automatically, 0 mainline delay)!
    Train->>VLM: Roof 3D laser scans Span #812: detect broken dropper clip (+340mm stagger, 7.9mm wire thickness)
    HBD->>MILP: Train enters depot at 23:40 -> schedule 1.2mm skim cut on Axle #0942 (Confirm, saving 80,000 km rim life)
    VLM->>MILP: Dispatch catenary tower wagon to Span #812 in 00:30–04:30 window interlocked with rail grinding gang!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Fixed Lathe + Static HBD | 🟨 Tier 2: Prompt Wayside LLM | 🟦 Tier 3: Per-Site Wayside ML | 🟩 Tier 4: Axle-Kalman + Catenary `VLM` + `MILP` |
|---|---|---|---|---|
| **Mainline Disruptions / Month** | `19.2 / mo` ✕ | `17.5 / mo` ✕ | `6.8 / mo` | 🏆 **`0.4 / mo` (`-97.9%` vs. baseline)** |
| **False Emergency Train-Stop Rate** | `24.5%` ✕ (brake rub & sun glint) | `27.0%` ✕ | `8.4%` ✕ (no disc/TADS fusion) | 🏆 **`0.3%` (dual-IR journal/disc + `TADS` acoustic)** |
| **Wheelset Rim Lathe Waste** | `21.8%` ✕ (8.0mm deep spall cuts) | `19.4%` ✕ | `10.5%` | 🏆 **`1.6%` (speed-normalized `WILD` `1.2mm` skim cut)** |
| **Catenary / Rail Defect Escape** | `3.4%` ✕ (manual walk blind spots) | `4.8%` ✕ (>32°C tamping risk) | `1.1%` ✕ | 🏆 **`0.0%` (300 km/h roof `VLM` + buckling gate)** |
| **Monthly Delay, Wheel & Track TCO** | `$960,000 / mo` | `$1,080,000 / mo` | `$410,000 / mo` | **`$84,000 / mo` 🏆 (-91.3%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Railway Rolling-Stock WILD/HBD, Catenary VLM & Possession Sizer](../reports/enterprise-railway-rolling-stock-catenary-track-maintenance-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-railway-rolling-stock-catenary-track-maintenance-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_railway_rolling_stock_catenary_track_maintenance_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_railway_rolling_stock_catenary_track_maintenance_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Link Every Wheelset Across All Wayside Detector Sites (`WILD / HBD / TADS`) via `AEI / RFID` Tags**: Track longitudinal Kalman degradation trajectories rather than triggering isolated single-site threshold alarms.
- [ ] **Cross-Check Infrared Journal Temperatures (`HBD`) Against Brake-Disc Pyrometers and Acoustic Bearing Arrays (`TADS`)**: Separate harmless sticking brake shoes from true tapered-roller bearing seizures before halting mainline traffic.
- [ ] **Normalize `WILD` Dynamic Wheel Impacts for Speed and Axle Load to Trigger Proactive `1.2mm` Skim Cuts**: Re-profile wheels on underfloor lathes before subsurface `RCF` cracks shell out into `8.0mm` deep defects.
- [ ] **Inspect Overhead Catenary Wire Thickness (`<8.2mm`) and Stagger (`>±300mm`) via Revenue-Train Roof `3D Laser + UV Arc VLMs`**: Catch dropper and contact-wire wear between manual walks.
- [ ] **Schedule 4-Hour Nightly Track Possessions via a Space-Time `MILP` Interlocked with Catenary Power Isolation and `32°C` Rail Buckling Thresholds**: Never disturb track ballast when rail steel exceeds stress-free neutral temperature limits.
