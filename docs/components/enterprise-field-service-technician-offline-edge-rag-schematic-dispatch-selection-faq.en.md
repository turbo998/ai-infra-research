# Enterprise Field Service Technician Copilot (`ServiceMax / Dynamics 365 Field Service / SAP FSM`), Offline-First Edge RAG (`3B/4B INT4 SLM + SQLite-Vec`), Exploded-View Schematic BOM Callout Pinning & Van-Stock Pre-Dispatch Selection Guide

> **Target Audience**: Cloud & AI Solution Architects, Edge/IoT Infrastructure Architects, and Field Service Operations Leaders building technician assistants for medical imaging (MRI/CT), HVAC/elevator fleets, industrial CNC machinery, wind turbines, and telecom towers (`ServiceMax`, `Microsoft Dynamics 365 Field Service`, `SAP FSM`, `Salesforce Field Service`).
> **Core Question**: When an enterprise operates **`8,000` field service technicians executing `180,000` monthly truck-roll work orders (`$450` cost per repeat second truck roll)**—with **`38.0%` of work sites inside underground hospital basements, factory Faraday cages, or remote substations with zero cellular signal**—why do naive "100% Cloud-Only LLM API Chatbots (Tier 1)" or "Text-Only Manual Chunking (Stripping Exploded-View Callout Balloons, Tier 2)" suffer **`38.0%` offline basement blackouts, `31.0%` wrong spare-part orders, and `26.5%` repeat truck rolls (`$380,000/month` in repeat visits and part returns)**? How does a **4-stage architecture combining Pre-Dispatch IoT Fault-to-Part Prediction + `Van-Stock` Inventory Matching + Morning Depot Wi-Fi `180MB` Offline Edge Sync (`4B INT4 On-Device SLM + SQLite-Vec + Serial As-Maintained BOM`) + Exploded-View Callout-to-BOM Pinning + Mandatory `LOTO` Safety Interlock Gate** boost First-Time Fix Rate (`FTFR`) from `73.5%` to **`95.8%`**, achieve **`100.0%` dead-zone availability**, cut wrong-part ordering to **`1.2%`**, and reduce monthly TCO by **`88.9%` (`$380,000 -> $42,000/month`)**?

![Enterprise Field Service Technician Offline Edge RAG, Schematic BOM Pinning & Van-Stock Dispatch Summary](../../assets/svg/enterprise-field-service-technician-offline-edge-rag-schematic-dispatch-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

In industrial and medical Field Service Management (`FSM`), **the most expensive failure is not an LLM taking 200ms longer—it is a technician driving 90 minutes to a hospital sub-basement only to find zero cellular signal to look up the wiring schematic, or discovering the failed solenoid valve isn't in their service van (`Van-Stock`), forcing an aborted visit and a `$450` second truck roll the next day**:

| Architecture Tier | Best-Fit Scenario | First-Time Fix Rate (`FTFR`) | Dead-Zone Offline Availability | Wrong Spare-Part Rate | Monthly Repeat Roll & Parts TCO | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 100% Cloud-Only API RAG + Blind Proximity Dispatch (Anti-Pattern)** | 100% urban 5G offices with zero physical spare-part replacements | `73.5%` (`26.5%` repeat rolls!) | **`62.0%` (`38%` blackout!)** | `18.4%` | `$380,000 / mo` | ❌ **Reject**: 100% unusable in MRI basements; massive repeat truck rolls |
| **Tier 2: Text-Only Manual Chunks (Strips CAD Exploded-View Callout Balloons)** | Simple text SOPs without mechanical assemblies or revision-specific BOMs | `76.0%` | `85.0%` (text cache only) | **`31.0%` (wrong P/N!)** | `$245,000 / mo` | ❌ **Reject**: Cannot map Callout `#14` on `DWG-882` to serial BOM |
| **Tier 3: Reactive On-Site Diagnostics Only (No Pre-Dispatch `Van-Stock` Check)** | Sites where 100% of spare parts are stored on-site at the customer facility | `80.5%` (`19.5%` missing part) | `95.0%` | `8.5%` | `$198,000 / mo` | ⚠️ **Incomplete**: Diagnoses fault on-site, but needed part isn't in the van |
| **Tier 4: Pre-Dispatch `Van-Stock` Check + Offline `4B SLM/SQLite-Vec` + Exploded-BOM Pinning + `LOTO` Gate (Recommended)** | **Enterprise Medical, Industrial, HVAC & Energy Field Service Production Standard** | **`95.8%`** | **`100.0%` (`420ms` on-device)** | **`1.2%`** | **`$42,000 / mo` (`-88.9%`)** | 🏆 **Production Gold Standard**: Right part in van before roll + 100% offline |

### 💡 Three Golden Rules for Field Service Architects
1. **Verify Spare-Part `Van-Stock` *Before* Rolling the Truck (`Pre-Dispatch Triage`), Not After Arriving On-Site**: When an equipment IoT alarm (`Error E-409: Hydraulic Pressure Drop`) or customer ticket arrives, the Pre-Dispatch Agent maps the asset's exact **`As-Maintained Serial BOM`** and predicts the top 2 likely replacement parts (`Valve P/N 8841-B [82%]`, `Seal Kit P/N 1092 [65%]`). The dispatch router **assigns a technician whose van inventory (`Van-Stock`) already carries those parts** (or schedules an automated locker pickup en route)—slashing repeat truck rolls from `26.5%` down to **`4.2%` (`95.8%` FTFR)**.
2. **Never Assume Cellular Connectivity On-Site: Pre-Sync a `180MB` Route-Scoped Offline Bundle Over Morning Depot Wi-Fi (`4B INT4 SLM + SQLite-Vec`)**: Don't try to cram `500GB` of enterprise manuals onto an iPad. When a technician's daily route (`6` work orders) is assigned at 07:00 AM over depot Wi-Fi, the mobile app automatically pre-syncs the **`SQLite-Vec` manual chunks, interactive SVG schematics, and `As-Maintained Serial BOM` for those specific 6 equipment serial numbers (`~180 MB` delta)** alongside a persistent on-device **`3B/4B INT4` SLM (`Llama-3.2-3B / Phi-4-Mini` on Apple MLX / Qualcomm NPU / ONNX Runtime)**—delivering **`100.0%` offline availability and `420ms` response latency in Sub-Basement B3**.
3. **Pin Exploded-View Callout Balloons (`#14`) Deterministically to the Asset's `As-Maintained Serial BOM` and Enforce a Mandatory `LOTO` Safety Interlock**: The same chiller model uses different solenoid valves in `2023 Rev A` vs. `2025 Rev C`. Tapping Callout Balloon `#14` on an exploded CAD drawing must query the specific unit's `As-Maintained Serial BOM` (`1.2%` wrong-part rate vs. `31.0%` in Tier 2), while high-voltage diagnostic steps remain locked until the technician completes the **Lockout/Tagout (`LOTO`) zero-energy verification gate**.

---

## 2. 🎯 Customer & Architect Critical Q&A

### Q1: Why does a "100% Cloud-Only GPT-4o / Claude Wrapper on 5G Tablets (Tier 1)" fail on `38.0%` of industrial/medical service calls and leave repeat truck rolls at `26.5%`?
* **Failure Mode 1: High-Value Equipment Lives Inside Faraday Cages and Sub-Basements (`38.0%` Dead Zones)**:
  * Hospital MRI rooms are encased in copper RF shielding, commercial HVAC chillers sit in Sub-Basement B3, semiconductor fabs block external cellular data, and wind turbines sit offshore or in rural dead zones. A cloud-only bot works in the parking lot and dies with `"Network Error"` right when the technician is kneeling in front of the open control cabinet.
* **Failure Mode 2: Diagnosing Only After Driving 90 Minutes On-Site Wastes `$450` When an `$80` Valve Is Missing from the Van**:
  * Proximity-only dispatch sends the nearest technician without checking whether their van carries the revision-specific replacement part for that fault code.

### Q2: Why does OCR Text-Only Manual Chunking (Tier 2) cause a `31.0%` wrong spare-part ordering rate?
* **Mechanical & Electrical Repairs Depend on Spatial Exploded-View Drawings (`Callout Balloon #14`) + Serial Engineering Change Orders (`ECO`)**:
  * Page 42 of a service manual shows an exploded assembly drawing with numbered balloons `①–⑥⑤`, while Page 45 lists part numbers conditioned on serial ranges and field retrofits (`ECO-992`). Text-only chunking severs the visual callout balloon from the serial-specific BOM table, causing the LLM to hallucinate or pick an obsolete revision's part number.

### Q3: How does Tier 4 fit thousands of equipment manuals onto a technician's tablet and handle offline work-order completion without database conflicts?
* **Route-Scoped Morning Wi-Fi Delta Sync (`180MB`) + `CRDT` Outbox Replay**:
  1. **Persistent Base Runtime (`~2.2 GB`)**: Quantized `Phi-4-Mini-3.8B-INT4` or `Llama-3.2-3B-INT4` weights (`2.0 GB`) + embedded `SQLite-Vec` engine stay installed on the tablet.
  2. **Daily Route Delta Bundle (`~180 MB / day`)**: Pulls vector chunks, vector SVG exploded drawings with clickable callout coordinates, and the exact `As-Maintained BOM` for the 6–8 asset serial numbers on the technician's 48-hour schedule.
  3. **Offline `CRDT` Outbox**: Work-order steps, `LOTO` photo hashes, `Van-Stock` barcode scans, and customer signatures are committed to a local encrypted SQLite outbox and automatically replayed to `ServiceMax / SAP ERP` once cellular connectivity returns.

---

## 3. 🧭 Visual Diagrams (Mermaid Decision Tree + Reference Architecture + Sequence Diagram)

### 3.1 Field Service Technician Copilot Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Field Service (FSM) Copilot Workload"] --> B{"Do technicians work in basements, shielded rooms, or remote sites?"}
    B -- "No (100% broadband & zero physical spare parts)" --> C["Cloud Multimodal RAG + Remote Video Assist"]
    B -- "Yes (Medical/Industrial/HVAC/Energy: 38% dead zones)" --> D["Enforce Pre-Dispatch Van-Stock + Offline Edge RAG (Tier 4)"]
    D --> E["Pre-Dispatch: IoT Fault Code -> Predict Top-2 Parts -> Match Technician Van-Stock"]
    D --> F["Morning Depot Wi-Fi Sync: 6 Assigned Asset Serials + SQLite-Vec + Exploded SVG (180MB)"]
    D --> G["On-Site Offline Edge: On-Device 4B INT4 SLM + LOTO Safety Gate + Callout #14 BOM Pinning"]
```

### 3.2 Tier 4 Pre-Dispatch Van-Stock + Offline Edge SLM + Exploded-BOM Reference Architecture

```mermaid
flowchart LR
    subgraph Cloud["☁️ Cloud Dispatch & Knowledge Hub (Pre-Dispatch & Depot Wi-Fi)"]
        IOT["Asset IoT Fault Telemetry\n(Error E-409)"] --> PRED["Probable Part Predictor\n(Queries As-Maintained Serial BOM)"]
        PRED --> DISP["Smart Dispatch Router (ServiceMax/D365)\nVerifies Technician Van-Stock Inventory"]
        DISP --> BUNDLE["Route-Scoped Offline Packager\n(SQLite-Vec + Vector CAD + Serial BOM ~180MB)"]
    end

    subgraph Edge["📱 Technician Rugged Tablet / iPad Offline Engine (Sub-Basement B3)"]
        SYNC["07:00 AM Depot Wi-Fi Delta Sync"] --> VEC["Local SQLite-Vec Index\n+ Serial As-Maintained BOM"]
        VEC --> LOTO["⚡ Mandatory LOTO Zero-Energy\nSafety Interlock Gate"]
        LOTO --> SLM["On-Device 3B/4B INT4 SLM\n(MLX / QNN / ONNX 420ms)"]
        CAD["Interactive Vector Exploded Diagram\nTap Callout Balloon #14 -> Exact P/N"] --> VEC
        SLM --> OUTBOX["Local Encrypted Outbox (CRDT)\nAuto-Syncs & Replenishes Parts on 5G"]
    end

    BUNDLE ==>|"Pre-Cached Before Truck Roll"| SYNC
```

### 3.3 End-to-End Pre-Dispatch, Offline Basement Repair & CRDT Sync Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Asset as Hospital B3 MRI Chiller (SN-9012)
    participant Dispatch as Cloud Pre-Dispatch Engine (ServiceMax)
    participant Tablet as Technician Tablet (4B INT4 SLM + SQLite-Vec)
    participant Tech as Field Service Technician

    Asset->>Dispatch: Emits IoT Alarm `E-409` (or Customer Ticket)
    Dispatch->>Dispatch: Inspects `SN-9012` As-Maintained BOM (`ECO-992` applied) -> Predicts `Valve P/N 8841-B (82%)`
    Dispatch->>Dispatch: Filters nearby technicians by `Van-Stock` -> Assigns Tech carrying `P/N 8841-B`
    Dispatch->>Tablet: 07:00 AM Depot Wi-Fi syncs 6-order bundle (`180MB`: manuals + `DWG-882` SVG + `SN-9012` BOM)
    Note over Tablet,Tech: 10:15 AM Tech enters Sub-Basement B3 RF-shielded room (Cellular Signal = 0 bars)
    Tech->>Tablet: Asks: "Fault E-409 with manifold chatter—what is pilot valve coil resistance?"
    Tablet->>Tech: 🛑 Triggers `LOTO` Interlock: "Lock out breaker CB-2 and confirm 0V reading before probing"
    Tech->>Tablet: Confirms LOTO zero-energy checklist
    Tablet->>Tablet: Local `SQLite-Vec` + `4B INT4 SLM` answers in `420ms` & highlights `DWG-882` Callout `#14`
    Tech->>Tablet: Taps Callout `#14` -> Locks exact `P/N 8841-B` & scans barcode from van stock
    Note over Tablet,Dispatch: 11:10 AM Tech exits basement -> `CRDT Outbox` syncs work order & triggers van restock
```

---

## 4. 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | Tier 1: 100% Cloud-Only API RAG + Blind Dispatch | Tier 2: Text-Only Manual Chunks (No Diagram Link) | Tier 3: On-Site Diagnosis Only (No Van-Stock Pre-Check) | Tier 4: Pre-Dispatch `Van-Stock` + Offline `4B SLM/SQLite-Vec` + Exploded-BOM Pinning (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **First-Time Fix Rate (`FTFR`)** | `73.5%` (`26.5%` repeat truck rolls) | `76.0%` | `80.5%` (`19.5%` missing van part) | **`95.8%` (Only `4.2%` heavy overhaul escalations)** |
| **Dead-Zone Offline Availability** | **`62.0%` (`38%` total blackout)** | `85.0%` (text search only) | `95.0%` | **`100.0%` (`420ms` on-device SLM + interactive CAD)** |
| **Wrong Spare-Part Order Rate** | `18.4%` | **`31.0%` (unlinked diagrams/ECOs)** | `8.5%` | **`1.2%` (Callout `#14` pinned to Serial BOM)** |
| **High-Voltage Safety Enforcement** | ❌ Unenforced prose | ❌ Unenforced prose | ⚠️ Passive banner | **✅ Mandatory `LOTO` interlock before test steps** |
| **Device Footprint** | 0 GB local (fails offline) | Unindexed PDFs | Bloated `>40GB` dump | **Compact `2.0GB` `4B INT4` base + `180MB` daily delta** |
| **Monthly Repeat Roll & Parts TCO** | `$380,000 / mo` | `$245,000 / mo` | `$198,000 / mo` | **`$42,000 / mo` (`-88.9%` savings)** |

---

## 5. 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Report

### 5.1 Baseline Workload (`8,000` Field Technicians / `180,000` Monthly Work Orders)
* **Workload Profile**: Global industrial and medical equipment fleet with `8,000` field technicians executing `180,000` monthly truck rolls (`38.0%` in zero-signal basements, shielded rooms, or remote sites), at `$450` fully loaded cost per repeat second truck roll.
* **Reproducible Model & Dataset**: See [`experiments/enterprise_field_service_technician_offline_edge_rag_schematic_dispatch_selection/run.py`](../../experiments/enterprise_field_service_technician_offline_edge_rag_schematic_dispatch_selection/run.py) and [`results.json`](../../experiments/enterprise_field_service_technician_offline_edge_rag_schematic_dispatch_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive Sizer**: 📊 [Open Enterprise Field Service Offline Edge RAG Sizer (`enterprise-field-service-technician-offline-edge-rag-schematic-dispatch-sizer.html`)](../reports/enterprise-field-service-technician-offline-edge-rag-schematic-dispatch-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`enterprise-field-service-technician-offline-edge-rag-schematic-dispatch-selection-cards.html`)](../reports/enterprise-field-service-technician-offline-edge-rag-schematic-dispatch-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Cloud-Only + Blind Dispatch", "ftfr_pct": 73.5, "offline_uptime_pct": 62.0, "wrong_part_pct": 18.4, "monthly_tco_k": 380.0 },
      { "tier": "Tier 2: Text-Only Manual Chunks", "ftfr_pct": 76.0, "offline_uptime_pct": 85.0, "wrong_part_pct": 31.0, "monthly_tco_k": 245.0 },
      { "tier": "Tier 3: On-Site Diagnosis Only", "ftfr_pct": 80.5, "offline_uptime_pct": 95.0, "wrong_part_pct": 8.5, "monthly_tco_k": 198.0 },
      { "tier": "Tier 4: Pre-Dispatch + Offline 4B SLM + BOM Pin", "ftfr_pct": 95.8, "offline_uptime_pct": 100.0, "wrong_part_pct": 1.2, "monthly_tco_k": 42.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "ftfr_pct": "Percentage",
    "offline_uptime_pct": "Percentage",
    "wrong_part_pct": "Percentage",
    "monthly_tco_k": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 95.8% First-Time Fix Rate and Cuts Monthly Repeat Truck-Roll TCO by 88.9%",
    "subtitle": "Monthly repeat truck-roll and wrong spare-part return TCO ($ USD Thousands) across 180K work orders/mo",
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

- [ ] **Pre-Dispatch IoT Fault-to-Part Prediction & `Van-Stock` Inventory Gate**: Does `ServiceMax / Dynamics 365 Field Service` predict probable replacement parts from the asset's `As-Maintained Serial BOM` and verify technician van inventory *before* dispatching the truck?
- [ ] **Morning Depot Wi-Fi Route-Scoped Delta Sync (`<250MB`)**: Does the tablet pre-cache `SQLite-Vec` chunks, interactive vector schematics, and serial BOMs for the next 48 hours of assigned work orders automatically over depot Wi-Fi?
- [ ] **Seamless On-Device `3B/4B INT4 SLM` Offline Failover**: Does the client switch to local `Llama-3.2-3B / Phi-4-Mini INT4` inference within `<50ms` when cellular RTT exceeds `800ms` or drops to zero bars?
- [ ] **Exploded-View Callout Balloon (`#14`) to Serial BOM Pinning**: Does tapping a callout balloon on an engineering drawing query the exact `As-Maintained Serial BOM` (accounting for `ECO` retrofits) rather than letting the LLM guess part numbers from text?
- [ ] **Mandatory Lockout/Tagout (`LOTO`) Safety Interlock Gate**: Are high-voltage, pressurized hydraulic, and laser diagnostic steps gated behind a mandatory `LOTO` zero-energy confirmation step in the tablet UI?
