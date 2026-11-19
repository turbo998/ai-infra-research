# Enterprise Automotive Aftermarket, Dealership Service Network (`DMS`), Intermittent Spare-Parts Inventory (`Croston / TSB` Multi-Echelon Pooling), Collision Photo `VLM` Estimating & Service-Bay Technician Scheduling `MILP` Agent Architecture Selection Guide: Legacy `DMS` Moving-Average + Manual Tear-Down + FIFO Bays vs. Prompt-Only Photo VLM vs. Retail Gaussian ML + 2D Mask vs. `TSB` Multi-Echelon Pooling + 17-Digit `VIN EPC` 3D Hidden-Damage `VLM` + Parts-Gated Bay `MILP` Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-automotive-aftermarket-parts-croston-collision-vlm-bay-milp-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Automotive OEM Aftermarket Supply-Chain VPs, Dealership Group Fixed-Operations Directors (`CDK Global / Reynolds & Reynolds / Tekion DMS`), and Collision MSO / InsurTech (`CCC / Audatex / Mitchell`) Leads.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection/results.json`](../../experiments/enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection/results.json).

![Enterprise Automotive Aftermarket Parts TSB, 17-Digit VIN Collision VLM & Bay MILP Summary](../../assets/svg/enterprise-automotive-aftermarket-parts-croston-collision-vlm-bay-milp-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating an automotive aftermarket network of **180 dealership service centers and 4 regional Parts Distribution Centers (`PDCs`) stocking `145,000` OEM part SKUs (`78%` of which are intermittent slow-movers selling only 0–3 units/year) across `2,800` daily service and collision repair orders (`ROs`)** exposes three severe operational failure modes:
1. **Moving-Average `DMS` Rules and Retail Gaussian ML Fail on `78%` Intermittent Slow-Moving SKUs (`18.4%` Vehicle-Off-Road `VOR` Stockouts Alongside `21.6%` Dead-Stock Obsolescence)**: Standard moving averages reorder immediately after a freak one-off part sale and watch the replacement sit unsold for 3 years, while classic Croston fails to decay demand probability during long zero-sales streaks!
2. **2D Collision Photo Estimators Miss Sub-Surface Hidden Structural/Sensor Damage (`26.5%` Post-Teardown Supplement Rework) and Order Wrong Trim Parts (`9.6%` Fitment Escape Without 17-Digit `VIN` Build-Sheet Gating)**.
3. **FIFO Service-Bay Scheduling Puts Vehicles on Lifts Before Parts Arrive, Blocking `$1,200/Day` Revenue Bays for 48+ Hours Waiting for Backordered Brackets**.

Evaluated across **180 dealerships, 4 regional PDCs, 145,000 OEM part SKUs, and 2,800 daily repair orders**, the four automotive aftermarket architectures compare as follows:

| Architectural Tier | Core Stack Combination | `VOR` Parts Stockout Rate | Dead-Stock Write-Off | Collision Supplement Rework | Wrong-`VIN` Part Fitment | Monthly Parts, Obsolescence & Bay Loss TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: `DMS` Moving-Avg + Manual Teardown** | 6-mo moving-average min/max + post-teardown supplements + FIFO lifts | `18.4%` ✕ | `21.6%` ✕ *(blind reorders)* | `26.5%` ✕ *(hidden damage surprise)* | `4.2%` ✕ *(missed supersessions)* | `$840,000 / mo` ✕ | ⛔ High VOR stockouts, dead stock & stalled lifts |
| **🟨 Tier 2: Prompt-Only Photo `VLM` & Parts Bot** | Feeding crash photos into a general VLM to guess part numbers & hours | `15.8%` ✕ | `19.4%` ✕ | `31.2%` ✕ *(misses inner impact bar)* | `9.6%` ✕ *(Matrix LED vs Halogen)* | `$950,000 / mo` ✕ | ⛔ Orders wrong trim parts & stalls bays |
| **🟦 Tier 3: Retail Gaussian ML + 2D Photo Mask** | Continuous retail ARIMA/XGBoost + 2D segmentation without VIN EPC AST | `7.4%` ✕ | `11.2%` ✕ *(no zero-period decay)* | `12.8%` ✕ *(surface-only estimate)* | `1.9%` ✕ | `$380,000 / mo` ✕ | ⚠️ Fails on 78% intermittent SKUs & hidden damage |
| **🟩 Tier 4: `TSB` Pooling + 17-Digit `VIN VLM` + Bay `MILP` Agent** | **`TSB` Intermittent Decay + `PDC` Lateral `MILP` + 3D Hidden-Damage `VLM` + 17-Digit `VIN EPC AST` + Parts-Gated Bay `MILP`** | **`0.6%` 🏆** *(-96.7%)* | **`1.8%` 🏆** *(-19.8 pts)* | **`1.9%` 🏆** *(-24.6 pts)* | **`0.0%` 🏆** *(Zero wrong part)* | **`$74,000 / mo` 🏆** *(-91.2%)* | 🌟 **Recommended Automotive Aftermarket Standard** |

> 💡 **Golden Rule for Automotive Aftermarket & Fixed-Operations AI Architects**: **"Never forecast intermittent spare parts (`78%` zero-demand weeks) with continuous retail ML or let a photo VLM guess OEM part numbers without a 17-digit `VIN` build sheet—model slow movers with Teunter-Syntetos-Babai (`TSB`) probability decay pooled at regional `PDCs` via 4-hour lateral transshipment `MILPs`, combine monocular 3D deformation `VLMs` with teardown Bayesian graphs to pre-order hidden impact bars and radar brackets, lock exact part revisions in the `VIN` As-Built `EPC AST`, and never assign a vehicle to a lift until 100% of its parts are scanned into the dealership bin."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Automotive Aftermarket & Collision Pain Points)

### Q1: Why do Moving Averages and Classic Croston fail on `78%` of automotive spare parts, and how does `TSB` + Multi-Echelon Pooling cut both stockouts (`0.6%`) and dead stock (`1.8%`)?
- **First-Principles Reality**:
  - Classic Croston updates demand interval estimates *only when a sale occurs*—if an aging vehicle model's part experiences 18 consecutive months of zero sales, Croston's forecast stays frozen at its old positive rate, causing massive dead-stock write-offs (`21.6%`).
  - Tier 4 uses **Teunter-Syntetos-Babai (`TSB`)** to update demand probability $\hat{p}_t$ **every single period** (decaying toward zero during dry spells) alongside conditional demand size $\hat{z}_t$:
    $$\hat{p}_t = \hat{p}_{t-1} + \beta \cdot \Big(\mathbb{I}(D_t > 0) - \hat{p}_{t-1}\Big), \quad \hat{z}_t = \hat{z}_{t-1} + \alpha \cdot (D_t - \hat{z}_{t-1}) \;\;(\text{if } D_t > 0)$$
  - Ultra-slow SKUs ($\hat{p}_t < 0.04$/wk) are pooled centrally at the 4 regional `PDCs` and dispatched via a **4-Hour Inter-Dealer Lateral Transshipment `MILP`** (`0.6%` VOR stockout, `1.8%` dead stock).

### Q2: How do the 3D Depth Hidden-Damage `VLM` and 17-Digit `VIN EPC AST` slash post-teardown supplement delays from `26.5%` to `1.9%` with `0.0%` wrong-trim parts?
- **Sub-Surface Bayesian Teardown Graph + Exact As-Built `PR / RPO` Option Resolution**:
  1. **Hidden Sub-Surface Damage Prediction**: Plastic bumper fascias rebound after impact, hiding crushed aluminum impact bars, energy absorbers, and blind-spot radar brackets underneath. Tier 4 fuses monocular 3D indentation depth (`mm`) with a **Conditional Teardown Probability Graph** trained on historical `CCC / Audatex` teardowns—pre-ordering hidden structural/sensor parts with $P \ge 0.85$ before the car ever enters the bay (`1.9%` supplement rework).
  2. **17-Digit `VIN` As-Built `EPC AST` & Supersession DAG**: A 2024 SUV has 6 visually similar headlamp assemblies (`Halogen` vs. `PR-8IT Matrix Laser LED`). Tier 4 decodes the 17-digit `VIN` against the OEM As-Built `PR/RPO` build sheet and traverses the **OEM Supersession DAG** (automatically bundling mandatory adapter harnesses when `Part A -> Part C`), achieving **`0.0%` wrong-part fitment**.

### Q3: How does the Parts-Gated Service-Bay `MILP` prevent disassembled vehicles from blocking `$1,200/day` hydraulic lifts?
- **Bin-Scan Readiness Interlock + High-Voltage (`HV`) Technician Skill Matching**:
  - The workshop `MILP` enforces a hard **Parts-in-Bin Gate**: scheduled collision and maintenance jobs are assigned a physical lift and technician slot *only after* 100% of VIN-verified parts (including predicted hidden-damage kits) are barcode-scanned into the dealership's staging bin, while matching `800V EV` battery jobs to `HV Level-3` certified technicians and ADAS calibration frames.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + TSB/VIN-VLM/Bay Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Automotive Aftermarket Parts, Collision Estimating & Dealership Bays<br/>(180 Dealerships + 4 PDCs + 145,000 SKUs + 2,800 Daily Repair Orders)"] --> B{"Uses DMS moving-average min/max + post-teardown supplements<br/>or prompt-only VLM photo part guessing?"}
    B -- "Yes (Moving-avg or prompt-only VLM)" --> C["⛔ High VOR Stockouts, Dead Stock & Stalled Lifts (Tier 1 / Tier 2)!<br/>18.4% stockouts, 21.6% dead-stock write-off, 26.5% teardown rework & 9.6% wrong parts"]
    B -- "Uses ML forecasting & computer vision" --> D{"Combines TSB intermittent pooling, 3D hidden-damage VLM,<br/>17-digit VIN EPC AST & parts-gated bay MILP?"}
    D -- "No (Retail Gaussian ML + 2D surface mask)" --> E["⚠️ Slow-Mover Misses & Hidden Teardown Surprises (Tier 3)<br/>7.4% stockouts, 11.2% dead stock & 12.8% supplement delays on hidden brackets"]
    D -- "Yes (TSB Pooling + VIN Hidden-Damage VLM + Bay MILP)" --> F["🟩 Tier 4: TSB Pooling + 17-Digit VIN VLM + Parts-Gated Bay MILP 🏆<br/>(0.6% Stockout | 1.8% Dead Stock | 1.9% Rework | 0.0% Wrong Part | -91.2% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `TSB` Multi-Echelon Pooling + 17-Digit `VIN` Hidden-Damage `VLM` + Bay `MILP` Topology

```mermaid
flowchart LR
    subgraph Inv["1. 145K-SKU TSB Intermittent Pooling"]
        DMS["180-Dealer Sales Stream<br/>(78% Intermittent SKUs)"] --> TSB["📦 TSB Probability-Decay Model (p_t * z_t)<br/>Decays Zero-Sales SKUs + Pools Slow Movers<br/>at 4 PDCs w/ 4h Lateral Transfer MILP"]
    end

    subgraph VLM_EPC["2. 3D Hidden-Damage VLM & 17-Digit VIN AST"]
        Photo["📸 Crash Photos + 17-Digit VIN"] --> VLM["🔍 Monocular 3D Depth + Teardown Graph<br/>Predicts Hidden Impact Bar & Radar (1.9% Rework)"]
        VLM --> EPC["🔒 OEM As-Built PR/RPO + Supersession AST<br/>Locks Exact Part & Harness Kit (0.0% Error)"]
    end

    subgraph Bay["3. Parts-in-Bin Gated Service-Bay MILP"]
        TSB & EPC --> Bin["✅ 100% Parts-in-Bin Barcode Gate"]
        Bin --> Lift["🔧 Lift + HV-Cert Tech + ADAS Frame MILP<br/>(Zero Lifts Blocked by Missing Parts)"]
    end
```

### 3.3 End-to-End Sequence: Pre-Ordering Hidden Collision Parts via 17-Digit `VIN` & Scheduling a Lift

```mermaid
sequenceDiagram
    autonumber
    participant Owner as Vehicle Owner (#RO-7712, 2024 SUV)
    participant VLM as 3D Hidden-Damage VLM & 17-Digit VIN EPC Engine
    participant PDC as Regional PDC TSB Pool & Lateral Transfer MILP
    participant Bay as Dealership Parts Bin & Service-Bay MILP

    Owner->>VLM: Upload 6 rear-collision smartphone photos + 17-digit VIN (visible plastic bumper crease)
    VLM->>VLM: 1. Compute 3D indentation depth -> Teardown Bayesian graph flags hidden impact bar (p=0.94) & radar bracket (p=0.91)!
    VLM->>VLM: 2. Decode 17-digit VIN As-Built PR codes (PR-8IT Matrix LED, PR-7Y1 Side Assist) + supersession DAG -> lock exact OEM part IDs (0.0% error)
    VLM->>PDC: Dispatch fascia + hidden impact bar + radar bracket from Regional PDC on 4-hour transfer shuttle
    PDC-->>Bay: Tue 18:00: All 5 parts physically barcode-scanned into Dealership Bin #14 -> emit `PARTS_100_PCT_READY`!
    Bay->>Owner: ⚡ Confirm Wed 09:00 slot on ADAS-capable Lift #14 with Master Tech -> single-visit repair with zero teardown hold!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: `DMS` Moving-Avg + FIFO | 🟨 Tier 2: Prompt Photo `VLM` Bot | 🟦 Tier 3: Retail Gaussian ML + 2D | 🟩 Tier 4: `TSB` + `VIN VLM` + Bay `MILP` |
|---|---|---|---|---|
| **`VOR` Parts Stockout Rate** | `18.4%` ✕ | `15.8%` ✕ | `7.4%` ✕ (lumpy demand miss) | 🏆 **`0.6%` (`TSB` intermittent model + `PDC` lateral transfer)** |
| **Dead-Stock Write-Off Rate** | `21.6%` ✕ (blind post-spike reorders) | `19.4%` ✕ | `11.2%` ✕ (no zero-sales decay) | 🏆 **`1.8%` (`TSB` probability decay + regional slow-mover pool)** |
| **Collision Supplement Rework** | `26.5%` ✕ (post-teardown surprises) | `31.2%` ✕ | `12.8%` ✕ (2D surface mask only) | 🏆 **`1.9%` (3D depth deflection + teardown Bayesian graph)** |
| **Wrong-`VIN` Part Fitment** | `4.2%` ✕ (manual catalog lookup) | `9.6%` ✕ (trim/headlamp confusion) | `1.9%` ✕ | 🏆 **`0.0%` (17-digit `VIN` As-Built `PR/RPO` + supersession `AST`)** |
| **Monthly Parts & Bay Loss TCO** | `$840,000 / mo` | `$950,000 / mo` | `$380,000 / mo` | **`$74,000 / mo` 🏆 (-91.2%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Automotive Aftermarket Parts (TSB), VIN Collision VLM & Bay Sizer](../reports/enterprise-automotive-aftermarket-parts-croston-collision-vlm-bay-milp-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-automotive-aftermarket-parts-croston-collision-vlm-bay-milp-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection/run.py --check
  python -m unittest tests/test_enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Replace Moving-Average `DMS` Rules and Continuous Gaussian ML with `Teunter-Syntetos-Babai (TSB)` on Intermittent SKUs**: Decay demand probability $\hat{p}_t$ during every zero-demand week and pool ultra-slow parts at regional `PDCs`.
- [ ] **Gate All Automated Part Orders on 17-Digit `VIN` As-Built `PR / RPO` Build Sheets and `EPC` Supersession DAGs**: Never order trim-sensitive components (`headlamps, radars, harnesses`) from visual photo similarity alone.
- [ ] **Pair Monocular 3D Collision Deformation `VLMs` with Historical Teardown Conditional Probability Graphs**: Pre-stage high-probability sub-surface structural and sensor parts before physical teardown begins.
- [ ] **Enforce a Physical `100% Parts-in-Bin Barcode Scan` Gate in the Service-Bay `MILP`**: Never put a non-emergency vehicle onto a lift until all required parts are physically received and scanned.
- [ ] **Couple Regional Weather Alerts (Hailstorms, Black Ice) to `PDC` Pre-Positioning Transfers**: Pre-stage windshields, bumper covers, and headlamps 48 hours ahead of severe weather events.
