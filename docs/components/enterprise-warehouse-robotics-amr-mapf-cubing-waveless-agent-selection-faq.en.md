# Enterprise Autonomous Warehouse Robotics (`AMR / Goods-to-Person AS/RS`), Multi-Agent Path Finding (`MAPF` Conflict-Based Search `CBS / PIBT`), 3D Bin-Packing Cubing `MILP` & Waveless Order-Release Agent Architecture Selection Guide: Discrete 4-Hour Batch Waves + Greedy `A*` + Volume-Sum Box Selection vs. Prompt-Only Task/Box LLM vs. Decentralized `MARL` Router vs. Windowed `MAPF` (`RHCR CBS / PIBT`) + Continuous Waveless `MILP` + 3D Extreme-Point Cubing Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-warehouse-robotics-amr-mapf-cubing-waveless-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, VPs of Supply-Chain & Warehouse Automation, Warehouse Execution System (`WES / RCS`) Principal Engineers, and Fulfillment Robotics Teams.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_warehouse_robotics_amr_mapf_cubing_waveless_agent_selection/results.json`](../../experiments/enterprise_warehouse_robotics_amr_mapf_cubing_waveless_agent_selection/results.json).

![Enterprise Warehouse Robotics (AMR), MAPF, 3D Cubing & Waveless Agent Summary](../../assets/svg/enterprise-warehouse-robotics-amr-mapf-cubing-waveless-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating a high-density **Goods-to-Person (`G2P`) fulfillment center with 800 Autonomous Mobile Robots (`AMRs`), 32 pick stations, and 160,000 daily shipped parcels (`$7.80` average Dimensional-Weight `DIM` shipping cost/parcel)** exposes three severe throughput and cost bottlenecks:
1. **Independent Greedy `A*` Routing Causes `18.5%` Narrow-Aisle Nose-to-Nose Deadlocks**: When 800 AMRs plan paths independently via standard `A*` (Tier 1) or unverified decentralized reinforcement learning (`MARL`, Tier 3) without a shared space-time vertex/edge reservation table, northbound and southbound robots enter opposite ends of single-lane storage aisles and lock bumpers—halting 18 queueing robots for 25 minutes until a technician manually intervenes!
2. **Discrete 4-Hour Batch Waves Starve Human Pickers for 35 Minutes at Every Wave Tail (`24.0%` Idle Rate, `185 UPH`)**: Waiting for the last 4 slow straggler pods of Wave 1 to finish before releasing Wave 2 leaves 28 of 32 pick stations standing idle.
3. **Scalar Volume-Sum Carton Selection Ships `34.0%` Void Air & Crushes Fragile Goods**: Summing scalar item volumes ($L \times W \times H = 11\text{L}$) to select a `30L` box (`40×30×25 cm`) fails whenever an order includes a `62 cm` tennis racket that physically cannot fit inside a `40 cm` box—or places a `2.5 kg` cast-iron skillet on top of a glass olive-oil bottle!

Evaluated across an **800-AMR, 32-station fulfillment center shipping 160,000 parcels/day**, the four architectural tiers compare as follows:

| Architectural Tier | Core Stack Combination | Station Pick Rate (`UPH`) | `AMR` Aisle Deadlock / Congestion | Pick-Station Starvation Idle | Carton `DIM` Void-Air Waste | Monthly Labor & `DIM` Shipping Loss | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: 4h Batch Wave + Greedy `A*` + Vol-Sum** | 4-hour batch waves + independent `A*` robot routing + scalar volume-sum box selection | `185 UPH` ✕ | `18.5%` ✕ *(aisle gridlock)* | `24.0%` ✕ *(wave-tail starvation)* | `34.0%` ✕ *(ships air)* | `$790,000 / mo` ✕ | ⛔ Half of robot & picker capacity wasted |
| **🟨 Tier 2: Prompt-Only Task & Box LLM** | Prompting an LLM to assign pods and choose box sizes from text descriptions | `165 UPH` ✕ | `22.0%` ✕ | `26.5%` ✕ | `37.5%` ✕ *(racket won't fit)* | `$910,000 / mo` ✕ | ⛔ Zero 3D geometry or deadlock guarantees |
| **🟦 Tier 3: Decentralized `MARL` Fleet Router** | Multi-agent RL routing without formal space-time reservation tables or 3D crush gates | `310 UPH` | `5.8%` ✕ *(intersection lock)* | `9.4%` | `19.2%` ✕ | `$340,000 / mo` ✕ | ⚠️ Gridlocks under viral 'hot-pod' surges |
| **🟩 Tier 4: Windowed `MAPF` (`CBS/PIBT`) + Waveless `MILP` + 3D Cubing Agent** | **Rolling-Horizon `CBS + PIBT` Space-Time Reservation + Waveless Little's-Law `MILP` + 3D Extreme-Point Cubing + Dynamic Split-Slotting** | **`495 UPH` 🏆** *(+167.6%)* | **`0.0%` 🏆** *(Provably deadlock-free)* | **`1.1%` 🏆** *(6-pod buffer)* | **`6.4%` 🏆** *(-27.6 pts)* | **`$68,000 / mo` 🏆** *(-91.4%)* | 🌟 **Recommended Architecture** |

> 💡 **Golden Rule for Warehouse Robotics & WES Architects**: **"Never route dense AMR fleets with independent single-agent `A*` and never release orders in rigid 4-hour batch waves—plan collision-free trajectories on a directed Space-Time Graph ($G \times T$) via Rolling-Horizon `MAPF` (`Windowed CBS + PIBT`, `0.0%` deadlock), govern continuous waveless order release via Little's Law to maintain a constant 6-pod buffer per station (`495 UPH`), and pack multi-item parcels via 3D Extreme-Point `MILP` with diagonal-fit and crush-weight constraints (`6.4%` void air)."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Real-World Warehouse Robotics & WES Pain Points)

### Q1: How does Rolling-Horizon Multi-Agent Path Finding (`RHCR Windowed CBS + PIBT`) guarantee `0.0%` narrow-aisle deadlocks across 800 AMRs at 50ms replanning speed?
- **First-Principles Reality**:
  1. **Why Independent `A*` and Unverified `MARL` Deadlock**: In a single-lane storage aisle (`B12`), if northbound `AMR-104` and southbound `AMR-319` enter opposite ends simultaneously, neither can pass or back up past queueing robots behind them.
  2. **Directed Space-Time Reservation Table ($G \times T$) + `PIBT`**: Tier 4 maintains a rolling 10-second **Space-Time Reservation Table** prohibiting both **Vertex Conflicts** ($\text{loc}(i,t) == \text{loc}(j,t)$) and **Edge/Corridor Swap Conflicts**. Using **Priority Inheritance with Backtracking (`PIBT`)** inside a windowed Conflict-Based Search (`CBS`) loop every **50ms**, higher-priority loaded `AMR-104` reserves the aisle corridor while `AMR-319` is held in a cross-aisle pull-out bay `(x=12, y=3)` for `2.5 seconds`—achieving **`0.0%` fleet deadlocks**.

### Q2: How do Continuous Waveless Order Release (Little's Law) and Dynamic Split-Slotting boost pick rates from `185 UPH` to `495 UPH`?
- **Eliminating Wave-Tail Starvation & "Hot-Pod" Contention**:
  - **Little's Law Waveless Buffer ($L = \lambda W$)**: Instead of waiting for a 4-hour batch wave to drain to zero, Tier 4 maintains an invariant **6-pod physical queue** at each of the 32 pick stations (`1.1%` starvation vs. `24.0%`). Whenever a pod departs a station, the waveless `MILP` injects a new order from the backlog that maximizes **Pile-On Hit Density** against the 6 pods already en route (`3.8 picks/pod`, rising to `5.2 picks/pod` on Cyber Monday).
  - **Dynamic Split-Slotting for Viral SKUs**: When a viral flash-sale item spikes `>4×` baseline velocity, Tier 4 decants inventory across **8 pods in distinct warehouse zones** so 18 stations never queue behind a single "hot pod".

### Q3: Why does scalar volume-sum carton selection fail on a `62 cm` tennis racket + `2.5 kg` skillet + glass oil bottle, and how does 3D Extreme-Point Cubing cut void air to `6.4%`?
- **3D Orthogonal/Diagonal Geometry + Z-Axis Crush Protection**:
  - Summing scalar volumes ($11\text{L} < 30\text{L}$) selects `Box #B2 (40×30×25 cm)`, whose maximum internal diagonal (`55.9 cm`) cannot fit a `62 cm` tennis racket! Tier 4 solves a **3D Extreme-Point Bin-Packing `MILP`** that verifies 3D diagonal fit inside `Box #B4 (55×35×20 cm, diagonal 65.1 cm)`, places the `2.5 kg` cast-iron skillet at `Z = 0`, and locks the glass olive-oil bottle upright with `<=1.0 kg` top-load (`6.4%` void air).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + MAPF/Waveless/Cubing Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Goods-to-Person (G2P) Robotics & WES<br/>(800 AMRs + 32 Pick Stations + 160,000 Parcels/Day)"] --> B{"Uses 4-hour batch waves + independent A* routing<br/>and scalar volume-sum box selection?"}
    B -- "Yes (Batch waves + greedy A* + vol-sum)" --> C["⛔ Aisle Gridlock & Picker Starvation (Tier 1 / Tier 2)!<br/>18.5% narrow-aisle deadlocks, 24% station idle time & 34% DIM void air"]
    B -- "Uses multi-agent fleet routing" --> D{"Combines Windowed MAPF (CBS/PIBT) space-time reservation,<br/>Waveless Little's-Law MILP & 3D Extreme-Point Cubing?"}
    D -- "No (Decentralized MARL without reservation table)" --> E["⚠️ Hot-Pod Congestion & Oversized Boxes (Tier 3)<br/>5.8% intersection gridlock & 19.2% void-air shipping waste"]
    D -- "Yes (Windowed MAPF + Waveless + 3D Cubing)" --> F["🟩 Tier 4: Windowed MAPF (CBS/PIBT) + Waveless WES + 3D Extreme-Point Cubing 🏆<br/>(495 UPH Pick Rate | 0.0% Deadlock | 1.1% Starvation | 6.4% Void Air)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Windowed `MAPF` + Continuous Waveless `WES` + 3D Extreme-Point Cubing Topology

```mermaid
flowchart LR
    subgraph WES["1. Continuous Waveless & Slotting Plane"]
        OMS["160,000 Daily Orders"] --> Release["🌊 Little's-Law Waveless MILP<br/>Maintains 6-Pod Station Buffer (1.1% Idle)<br/>Maximizes Pile-On Density (3.8 Picks/Pod)"]
        Release --> Slot["🔥 Dynamic Viral Split-Slotting<br/>Spreads >4x Velocity SKUs Across 8 Zones"]
    end

    subgraph Fleet["2. 800-AMR Windowed MAPF Fleet Plane"]
        Slot --> MAPF["🤖 RHCR Windowed CBS + PIBT Planner<br/>50ms Space-Time Graph G × T Reservation<br/>Locks Vertex & Corridor Swaps (0.0% Deadlock)"]
    end

    subgraph Pack["3. 3D Extreme-Point Cubing Plane"]
        MAPF --> Station["⚡ 32 High-Speed G2P Stations (495 UPH)"]
        Station --> Cube["📦 3D Extreme-Point Bin-Packing MILP<br/>Heavy-on-Bottom (Z=0) + Upright Liquid + Diagonal Fit<br/>(6.4% Carton Void Air)"]
    end
```

### 3.3 End-to-End Sequence: Resolving a Narrow-Aisle Conflict in 50ms & Packing a 3D Multi-Item Order

```mermaid
sequenceDiagram
    autonumber
    participant WES as Waveless Order-Release MILP
    participant MAPF as Windowed CBS/PIBT Fleet Planner (50ms)
    participant AMR as AMR-104 (North) & AMR-319 (South)
    participant Cube as 3D Extreme-Point Carton Cuber

    WES->>WES: Station #04 buffer drops to 5 pods -> release 3 backlog orders sharing SKUs on incoming pods (3.8 picks/pod)
    WES->>MAPF: Dispatch loaded AMR-104 North through single-lane Aisle B12 as empty AMR-319 approaches from the North
    MAPF->>MAPF: Detect t=12.0s corridor-swap conflict in G × T table -> PIBT holds AMR-319 in pull-out bay (x=12, y=3) for 2.5s
    MAPF->>AMR: ⚡ Stream conflict-free space-time trajectories (Act automatically) -> 0.0% aisle deadlock!
    AMR->>Cube: Deliver Order #ORD-88412 (62cm tennis racket + 2.5kg iron skillet + 500mL upright glass oil)
    Cube->>Cube: Reject 40cm Box #B2; solve 3D Extreme-Point MILP -> select Box #B4 (65.1cm diagonal), skillet at Z=0, upright bottle
    Cube-->>WES: ✅ Render 3D visual packing diagram at pack station (Confirm) with 6.4% void air!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: 4h Wave + Greedy `A*` | 🟨 Tier 2: Prompt Task/Box LLM | 🟦 Tier 3: Decentralized `MARL` | 🟩 Tier 4: Windowed `MAPF` + Waveless + 3D |
|---|---|---|---|---|
| **Station Pick Rate (`UPH`)** | `185 UPH` ✕ | `165 UPH` ✕ | `310 UPH` | 🏆 **`495 UPH` (`+167.6%`, waveless 6-pod buffer)** |
| **`AMR` Aisle Deadlock / Congestion** | `18.5%` ✕ (nose-to-nose gridlock) | `22.0%` ✕ | `5.8%` ✕ (hot-pod cycle locks) | 🏆 **`0.0%` (`RHCR CBS/PIBT` space-time reservation)** |
| **Pick-Station Starvation Idle** | `24.0%` ✕ (wave-tail stragglers) | `26.5%` ✕ | `9.4%` | 🏆 **`1.1%` (Little's-Law continuous waveless flow)** |
| **Carton `DIM` Void-Air Waste** | `34.0%` ✕ (scalar volume sum) | `37.5%` ✕ (crushes fragile items) | `19.2%` ✕ | 🏆 **`6.4%` (3D Extreme-Point + `Z=0` crush gate)** |
| **Monthly Labor & `DIM` Loss** | `$790,000 / mo` | `$910,000 / mo` | `$340,000 / mo` | **`$68,000 / mo` 🏆 (-91.4%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Warehouse Robotics (AMR), MAPF, 3D Cubing & Waveless Sizer](../reports/enterprise-warehouse-robotics-amr-mapf-cubing-waveless-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-warehouse-robotics-amr-mapf-cubing-waveless-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_warehouse_robotics_amr_mapf_cubing_waveless_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_warehouse_robotics_amr_mapf_cubing_waveless_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Enforce a Deterministic Space-Time Vertex and Edge-Swap Reservation Table ($G \times T$) via Rolling-Horizon `MAPF` (`CBS + PIBT`)**: Never allow AMRs to enter single-lane storage aisles using uncoordinated greedy `A*` paths.
- [ ] **Replace Discrete Batch Waves with Little's-Law Continuous Waveless Order Release**: Maintain a constant 6-pod queue per pick station while maximizing multi-order `Pile-On Hit Density`.
- [ ] **Trigger Dynamic Split-Slotting Across Multiple Warehouse Zones When SKU Velocity Exceeds `4×` Baseline**: Eliminate single-pod contention bottlenecks during flash sales.
- [ ] **Compute Carton Selection via 3D Extreme-Point Bin-Packing `MILP` Rather Than Scalar Volume Sums**: Enforce 3D diagonal fit, upright liquid orientation, and heavy-on-bottom ($Z = 0$) crush constraints.
- [ ] **Execute `<50ms` Local Subgraph Re-Planning Around Disabled AMRs or Blocked Fiducial Nodes**: Prevent single-robot faults from halting cross-highway traffic.
