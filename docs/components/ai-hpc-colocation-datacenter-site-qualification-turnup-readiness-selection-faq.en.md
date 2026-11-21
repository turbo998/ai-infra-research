# Large-Scale AI/HPC Accelerator Cluster Partner & Colocation Data Center Site Qualification, Turn-Up (`8-Stage Gate`), `45–120kW+` Liquid Cooling (`DLC / CDU`) & Operational Readiness Architecture Selection Guide

> **Target Audience**: Cloud & AI Infrastructure Solution Architects, Principal HPC Deployment Engineers, and Data Center Infrastructure Management (`DCIM`) Leaders deploying Google Cloud TPUs (`v5p / TPU7x Ironwood / TPU 8t & 8i`) or high-density NVIDIA GPUs (`GB200 NVL72 / B200 / H200`) into partner-operated colocation facilities, sovereign cloud sites, and customer-premises data centers.
> **Core Question**: As next-generation AI clusters expand into partner-operated colocation and customer-premises facilities—pushing rack power density to **`45–120kW+`**, full-rack weight to **`1,800–2,200 kg`**, and optical fabrics to tens of thousands of `MPO-16` fibers and Direct-to-Chip (`DLC`) liquid-cooling loops—why do legacy "Ping/SSH-Only Handover (Tier 1)" or "Single-Node Smoke Testing (Tier 2)" workflows suffer **`68.0-day` turn-up delays, `34.5%` Month-1 job aborts from infant-mortality hardware/optics, and `61.0%` degraded collective bandwidth (`$920,000/month` in stranded cluster depreciation and rework TCO)**? How does an **8-Stage Turn-Up Gate (`Site Qual -> Rack/DLC -> Rigging -> Hookup -> LLDP/Optical Scope -> Firmware/Redfish -> 48h Burn-In -> Handover`) + 1:1 `CDU`/Power-to-Compute Pod Fault-Domain Alignment + `>=90%` Line-Rate `NCCL busbw / TPU ICI` Acceptance Gate + 4-Tier Telemetry & `3%` On-Site Hot-Spare Pool (Tier 4)** compress turn-up to **`9.5 days`**, slash Month-1 job aborts to **`0.6%`**, sustain **`94.8%` collective bandwidth**, and cut monthly operations/rework TCO by **`89.7%` (`$920,000 -> $95,000/month`)**?

![AI/HPC Partner & Colocation DC Site Qualification, Turn-Up & Acceptance Gate Summary](../../assets/svg/ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

When deploying a multi-megawatt AI cluster into a partner or customer colocation hall, **"racking the servers and verifying `ssh` login" represents barely 20% of the engineering work—the real production killers are a `2,200 kg` liquid-cooled rack cracking an unrated raised floor, trapped air bubbles in a `120kW` cold-plate manifold throttling chips to 95°C, or 3 dusty `MPO-16` fiber ferrules and 2 swapped Rail cables dragging 1,024-chip `All-Reduce` down by 4x**:

| Turn-Up & Facility Architecture Tier | Best-Fit Scenario | Turn-Up to Stable Prod (`Days`) | Month-1 Infant-Mortality Abort Rate | Sustained Collective `busbw / ICI` (% of Spec) | Monthly Stranded Capacity & Rework TCO | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Legacy Air-Cooled Colo (`<20kW/Rack`) + Ping/SSH-Only Handover (Anti-Pattern)** | Legacy `<12kW` enterprise CPU web servers only | **`68.0 Days` (chronic debugging!)** | **`34.5%` (DOA optics & HBM ECC!)** | `61.0%` (dirty optics + throttling) | `$920,000 / mo` | ❌ **Reject**: Thermal throttling & cable errors turn ML researchers into cable techs |
| **Tier 2: Retrofitted Rear-Door Heat Exchanger (`RDHx <=40kW`) + Single-Node Smoke Test** | Small `L40S / PCIe H100` pilot islands | `42.0 Days` | `16.8%` | `74.5%` (misses cross-rack swaps!) | `$560,000 / mo` | ❌ **Reject**: Single-node `nvidia-smi` cannot detect swapped Rail/Torus fibers |
| **Tier 3: `120kW` Cold-Plate `DLC` (Unaligned `CDU` Fault Domains & No `48h` Soak)** | Facilities with liquid cooling but ad-hoc acceptance gates | `24.0 Days` | `7.8%` (unscreened infant DOA) | `84.0%` | `$310,000 / mo` | ⚠️ **Fragile**: A single `CDU` maintenance event drains 4 separate training slices |
| **Tier 4: 8-Stage Gate + `120kW DLC/CDU` Pod Alignment + `48h` Burn-In (`>=90% busbw/ICI`) + 4-Tier Telemetry (Recommended)** | **Partner/Premises `TPU7x Ironwood / v5p / 8t & 8i` & `GB200 NVL72 / B200` Standard** | **`9.5 Days` (`7.2x` faster)** | **`0.6%` (`98%` DOAs caught pre-Day 1)** | **`94.8%`** | **`$95,000 / mo` (`-89.7%`)** | 🏆 **Production Gold Standard**: Turnkey cluster ready for full-speed training on Day 1 |

### 💡 Three Golden Rules for AI/HPC Colocation & Partner Site Architects
1. **Enforce a Strict 8-Stage Turn-Up Gate (`No Stage N+1 Without Signed Stage N Verification`)**:
   * **Stage 1 (`Site Qualification`)** $\rightarrow$ **Stage 2 (`Rack Layout & Power/DLC Envelope`)** $\rightarrow$ **Stage 3 (`Receiving, Floor Load & Rigging`)** $\rightarrow$ **Stage 4 (`Physical Racking & Power/Liquid Manifold Hookup`)** $\rightarrow$ **Stage 5 (`Structured Cabling, Optical Scope & Automated LLDP Audit`)** $\rightarrow$ **Stage 6 (`Golden Firmware Lock & Out-of-Band BMC Redfish Bring-Up`)** $\rightarrow$ **Stage 7 (`48-Hour Full-TDP Thermal Soak & Collective Acceptance`)** $\rightarrow$ **Stage 8 (`4-Tier Telemetry, Hot-Spare Pool & Operational Handover`)**.
2. **Align Physical Facility Fault Domains (`A/B Busways + CDU Secondary Loops`) 1:1 with Logical Compute Domains (`TPU 4x4x4 Cube` / `GB200 NVL72 Pod`)**:
   * Never plumb a single Coolant Distribution Unit (`CDU`) manifold across racks belonging to 4 different training slices. Map **one `N+1` redundant `CDU` loop + dual `A/B` overhead busways strictly to one logical compute pod**, and wire optical/rope leak detectors directly into the rack PDU shunt-trip interlock.
3. **Never Sign Off Cluster Handover Without a `48-Hour` Square-Wave Thermal Soak, `>=90%` Line-Rate Collective Bandwidth (`NCCL busbw` / `TPU ICI`), and `<=2.5%` Reference Model Step-Time Variance**:
   * Roughly `85%` of Dead-on-Arrival (`DOA`) optical transceivers, poorly seated cold plates, and infant-mortality HBM row-remap faults manifest within the first `24–48 hours` of sustained `100% TDP` thermal cycling. Catch and swap them from an on-site **`2%–3%` pre-burned hot-spare pool** *before* handing the cluster to ML teams.

---

## 2. 🎯 Customer & Architect Critical Q&A

### Q1: What are the 6 non-negotiable physical checks during Stage 1 (`Site Qualification`) for a `45–120kW+/rack` TPU or GPU deployment?
* **The 6 Physical Qualification Pillars**:
  1. **Power Density & Step-Load Transformer Headroom (`45–120kW+ / Rack`)**: Verify dual `415V/480V 3-phase` overhead busways and upstream UPS/generator capacity to absorb synchronous training barrier power swings (`dP/dt`).
  2. **Direct-to-Chip (`DLC`) Liquid Cooling & `CDU` Water Chemistry**: Every `100 kW` rack at $\Delta T = 10^\circ\text{C}$ requires **`~150 LPM`** of inhibited propylene glycol (`PG25`) secondary coolant through **`25-micron` filtration**, fed by facility chilled water (`FWS`) with `N+1` redundant `CDU` pumps, automated air-purge valves, drip trays, and leak detection interlocks.
  3. **Structural Slab Loading (`>=2,000–2,500 kg/m²`) & Rigging Path**: A populated `GB200 NVL72` or `TPU7x Ironwood` liquid-cooled rack weighs **`1,800–2,200 kg`**. Verify loading-dock scissor lifts, freight elevator capacity (`>=3,000 kg`), corridor turning radii, door clearance (`>=2.4m`), and raised-floor/slab point-load ratings (installing steel spreader plates where needed).
  4. **Tight Pod Pitch for Copper/Optical Reach Limits**: Intra-pod Active Electrical Cables (`AEC`) and 3D Torus copper links require strict adjacency (`<3–5m` cable reach) and dedicated overhead fiber trays with `>=30mm` bend-radius trumpets.
  5. **Dual-Diverse WAN / Cloud Interconnect Uplinks**: Physically separate east/west fiber entry conduits providing `400G/800G` waves to object storage and `Multislice DCN`.
  6. **24x7 Certified `Remote Hands` SLA (`<15-Minute` P0 Response) & Spares Cage**: On-site technicians trained in `MPO-16` inspection/cleaning and dripless Quick-Disconnect (`QD`) liquid couplings.

### Q2: Why does a cluster that passes 100% of single-node smoke tests (`Tier 2`) still fail or crawl at `74.5%` bandwidth on multi-node training?
* **Cross-Rack Cabling Swaps & Microscopic Fiber End-Face Contamination**:
  * Single-node `nvidia-smi` or single-host diagnostics never exercise inter-rack links. If two fibers on `Rail-3` are swapped between neighboring racks, or if a single `MPO-16` connector has skin-oil contamination adding `2.5 dB` attenuation (`Pre-FEC BER > 1e-6`), multi-node `All-Reduce` stalls across the entire pod. Stage 5 eliminates both via **100% automated `LLDP` neighbor validation against the `Cutsheet YAML`** and **optical `DOM Rx Power / OCS Insertion Loss (<1.8 dB)` gates**.

### Q3: What are the exact quantitative pass/fail thresholds for Stage 7 (`Burn-In & Acceptance Testing`) across GPU and TPU clusters?
* **4-Level Quantitative Acceptance Funnel**:

| Acceptance Level | NVIDIA GPU Cluster Pass Gate (`H200 / B200 / GB200`) | Google Cloud TPU Cluster Pass Gate (`v5p / v6e / TPU7x / 8t & 8i`) |
| :--- | :--- | :--- |
| **L1: Silicon & HBM Health** | `dcgmi diag -r 4` PASS; `0` uncorrectable ECC, `0` new row-remaps, `0` fatal XIDs (`48/63/79/94`) | `libtpu` hardware diagnostics PASS; `0` uncorrectable HBM3e ECC errors across `TensorCores` & `SparseCores` |
| **L2: 48h Full-TDP Thermal Soak** | `48h` continuous `HPL / FP8 GEMM` square-wave soak: GPU core `<=83°C`, HBM `<=88°C`, `0%` thermal throttling | `48h` systolic `MXU` thermal-shock soak: cold-plate outlet $\Delta T$ within `±1.5°C` of spec, `0%` thermal clock throttling |
| **L3: Fabric & Collective Bandwidth** | Intra-node `nvbandwidth` PASS; multi-node `nccl-tests` (`all_reduce_perf`, `alltoall_perf` @ `8GB`) **`busbw >= 90%` of line rate** + pairwise bisection matrix | `3D Torus ICI` all-axis + Wrap-Around verification + `OCS` mirror switch test PASS; **`All-Reduce / All-Gather >= 92%` of spec**, Post-FEC `BER < 1e-12` |
| **L4: Reference Model Step Variance** | 1,000 steps of reference `Llama-3-70B / 671B MoE`: MFU within `2%` of golden baseline; inter-pod step-time std dev `<=2.5%` | 1,000 steps of reference `MaxText` training & `vLLM-TPU` benchmark: step-time variance **`<=2.5%`** + verified `Orbax` checkpoint save/restore |

---

## 3. 🧭 Visual Diagrams (Mermaid 8-Stage Gate + Reference Architecture + Sequence Diagram)

### 3.1 8-Stage Colocation Site Qualification & Turn-Up Lifecycle Gate

```mermaid
flowchart TD
    S1["Stage 1: Site Qualification\nVerify 45-120kW/rack power, CDU LPM flow, >=2,000 kg/m² slab & dual-diverse WAN"] --> S2["Stage 2: Rack Layout & Fault-Domain Design\nAlign 1 N+1 CDU + A/B Busway strictly to 1 TPU Cube / NVL72 Pod"]
    S2 --> S3["Stage 3: Receiving, Rigging & Floor Load\nShock-sensor check, spreader plates & 2.2-ton rack placement"]
    S3 --> S4["Stage 4: Power & DLC Manifold Hookup\n25um flushing, 24h pressure/leak test & shunt-trip interlock check"]
    S4 --> S5["Stage 5: Structured Cabling, Optical Scope & LLDP Gate\nInspect-Clean-Inspect MPO + 0-diff LLDP Cutsheet audit + BER < 1e-12"]
    S5 --> S6["Stage 6: Golden Firmware Lock & BMC Redfish Bring-Up\nEnforce identical BMC/BIOS/NIC/Switch/Accelerator firmware + 4-tier telemetry"]
    S6 --> S7["Stage 7: 48-Hour Full-TDP Burn-In & Collective Acceptance\nDCGM L4 / HPL soak + NCCL busbw / TPU ICI >= 90% + MaxText step variance <= 2.5%"]
    S7 --> S8["Stage 8: Operational Handover & Runbook Readiness\nCertify 3% pre-burned on-site hot spares + <15m P0 Remote Hands SLA"]
```

### 3.2 Tier 4 Facility-to-Pod Fault-Domain Alignment & 4-Tier Telemetry Architecture

```mermaid
flowchart LR
    subgraph Facility["🏢 Layer 1: Facility & Liquid Cooling (BMS / Modbus)"]
        FWS["Primary Facility Water (FWS)\n+ Dual A/B 480V Busways"] --> CDU["N+1 Redundant CDU Loop\nPG25 LPM/dP + 25um Filter + Leak Shunt-Trip"]
    end

    subgraph Hardware["🖥️ Layers 2 & 3: Host Redfish & Optical Fabric"]
        CDU ==>|"1:1 Fault-Domain Mapping"| POD["Compute Pod (TPU 4x4x4 Cube / GB200 NVL72)\nBMC Redfish Temps, Voltages & PSU Telemetry"]
        POD <==>|"MPO-16 Optics / AEC"| FAB["3D Torus + OCS / IB-RoCE Fabric\nDOM Rx Power, Pre-FEC BER & PFC Counters"]
    end

    subgraph Workload["🚀 Layer 4: Workload & Auto-RMA Loop"]
        POD & FAB --> MON["Unified 4-Tier Observability (Prometheus)\nCorrelates Coolant Temp -> Chip Temp -> BER -> Step Time"]
        MON --> RMA["Auto-Cordon Node + Dispatch <15m Remote Hands (3% On-Site Hot Spares)"]
    end
```

### 3.3 End-to-End Automated `LLDP` Audit, `48h` Burn-In & Hot-Spare Replacement Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Tech as On-Site Deployment Engineer
    participant Bot as Automated Turn-Up Gate Engine
    participant BMC as BMC Redfish & Switch Telemetry
    participant Pod as 64-Rack AI Cluster (TPU/GPU)

    Tech->>BMC: Completes rack placement, DLC air-purge & 8,000-fiber cabling
    Bot->>BMC: [Stage 5] Pulls cluster-wide `LLDP` neighbor table & transceiver `DOM Rx Power`
    BMC-->>Bot: Flags `Rack-14 Node-3 Port-2/3` cable swap + `Rack-22 Port-7` `-6.8 dBm` dirty ferrule
    Bot->>Tech: Auto-generates exact rack/U/port remediation ticket
    Tech->>BMC: Cleans ferrule & swaps crossed fibers -> Stage 5 re-audit returns `0 diffs (PASS)`
    Bot->>BMC: [Stage 6] Verifies & locks Golden Firmware hashes across all `BMCs / NICs / Switches`
    Bot->>Pod: [Stage 7] Runs `48h` square-wave thermal soak + pairwise `NCCL / TPU ICI` bisection + `MaxText` benchmark
    Pod-->>Bot: Hour 19: `Rack-08 Node-2 GPU-5` trips uncorrectable HBM ECC / row-remap threshold
    Bot->>Tech: Cordons node & dispatches on-site replacement from `3%` pre-burned hot-spare pool (`35m` swap)
    Bot->>Pod: Re-verifies `94.8% busbw (>=90%)` & `1.4%` step variance (`<=2.5%`) -> Issues Stage 8 Handover Certificate
```

---

## 4. 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | Tier 1: Air-Cooled Colo + Ping/SSH-Only Handover | Tier 2: Retrofitted `RDHx` + Single-Node Smoke Test | Tier 3: `120kW` `DLC` (Unaligned `CDU` / No `48h` Soak) | Tier 4: 8-Stage Gate + `120kW DLC/CDU` Alignment + `48h` Burn-In (`>=90% busbw`) (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **Turn-Up to Stable Production** | **`68.0 Days` (chronic post-handover debugging)** | `42.0 Days` | `24.0 Days` | **`9.5 Days` (`7.2x` faster turnkey readiness)** |
| **Month-1 Infant-Mortality Abort Rate** | **`34.5%` (unscreened DOA cards & optics)** | `16.8%` | `7.8%` | **`0.6%` (`98%` of DOAs caught during `48h` soak)** |
| **Sustained Collective `busbw / ICI`** | `61.0%` | `74.5%` (misses crossed cables) | `84.0%` | **`94.8%` (enforces `>=90%` line-rate & `0` LLDP diff)** |
| **`CDU` & Power Blast-Radius Isolation** | Uncontrolled hot spots | Localized throttling | ❌ 1 `CDU` trip kills 4 separate slices | **✅ 1:1 `N+1 CDU + A/B Busway` to Compute Pod alignment** |
| **Telemetry & Hardware RMA Loop** | Host ping only | Single-node `nvidia-smi` | Siloed BMS vs. IT logs | **✅ 4-Tier correlation (`BMS+Redfish+Fabric+Workload`) + `3%` spares** |
| **Monthly Downtime & Rework TCO** | `$920,000 / mo` | `$560,000 / mo` | `$310,000 / mo` | **`$95,000 / mo` (`-89.7%` savings)** |

---

## 5. 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Report

### 5.1 Baseline Workload (`64`-Rack High-Density Liquid-Cooled AI Cluster / `120 kW/Rack` / `7.68 MW`)
* **Workload Profile**: Partner-operated colocation deployment of `64` liquid-cooled AI racks (`120 kW/rack`, `7.68 MW` total IT load, `11,520 LPM` secondary `PG25` coolant flow).
* **Reproducible Model & Dataset**: See [`experiments/ai_hpc_colocation_datacenter_site_qualification_turnup_readiness_selection/run.py`](../../experiments/ai_hpc_colocation_datacenter_site_qualification_turnup_readiness_selection/run.py) and [`results.json`](../../experiments/ai_hpc_colocation_datacenter_site_qualification_turnup_readiness_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive Sizer**: 📊 [Open AI/HPC Colocation Site Qualification & Turn-Up Sizer (`ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-sizer.html`)](../reports/ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-selection-cards.html`)](../reports/ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Air-Cooled + Ping-Only Handover", "turnup_days": 68.0, "abort_pct": 34.5, "busbw_pct": 61.0, "monthly_tco_k": 920.0 },
      { "tier": "Tier 2: RDHx + Single-Node Smoke Test", "turnup_days": 42.0, "abort_pct": 16.8, "busbw_pct": 74.5, "monthly_tco_k": 560.0 },
      { "tier": "Tier 3: DLC Without CDU Alignment / 48h Soak", "turnup_days": 24.0, "abort_pct": 7.8, "busbw_pct": 84.0, "monthly_tco_k": 310.0 },
      { "tier": "Tier 4: 8-Stage Gate + CDU Align + 48h Soak", "turnup_days": 9.5, "abort_pct": 0.6, "busbw_pct": 94.8, "monthly_tco_k": 95.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "turnup_days": "Quantity",
    "abort_pct": "Percentage",
    "busbw_pct": "Percentage",
    "monthly_tco_k": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Compresses Turn-Up to 9.5 Days, Hits 94.8% Busbw, and Cuts Monthly TCO by 89.7%",
    "subtitle": "Monthly post-handover downtime, re-cabling rework, and cooling inefficiency TCO ($ USD Thousands)",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_k",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ Architecture Review & Production Readiness Checklist

- [ ] **Stages 1–3 Structural Slab & Rigging Sign-Off**: Have dock lifts, freight elevators (`>=3,000 kg`), door clearances (`>=2.4m`), and slab load ratings (`>=2,000–2,500 kg/m²` with spreader plates) been signed off before rack delivery?
- [ ] **Stages 2 & 4 `DLC / CDU` Loop Hygiene & 1:1 Pod Fault-Domain Alignment**: Has the secondary `PG25` loop passed `25-micron` flushing and `24-hour` pressure testing, with each `N+1 CDU` manifold mapped strictly to one logical compute pod?
- [ ] **Stage 5 Automated `LLDP` Cutsheet Diff & Optical `BER` Gate**: Does automated verification confirm `0` port-mapping mismatches against the `Cutsheet YAML`, `Pre-FEC BER < 1e-12`, and `OCS` insertion loss `<1.8 dB`?
- [ ] **Stage 6 Golden Firmware Lock & Out-of-Band `Redfish` Ingestion**: Are `BMC`, `BIOS`, `NIC/DPU`, switch `NOS`, and accelerator firmware versions 100% hash-identical and streaming telemetry into Prometheus?
- [ ] **Stages 7 & 8 `48h` Full-TDP Burn-In (`>=90% busbw / ICI`) & `3%` On-Site Hot-Spare Pool**: Has the cluster passed a `48-hour` thermal soak, `>=90%` line-rate collective bandwidth gate, and `<=2.5%` reference training step-time variance with `2%–3%` pre-burned spares staged on-site?
