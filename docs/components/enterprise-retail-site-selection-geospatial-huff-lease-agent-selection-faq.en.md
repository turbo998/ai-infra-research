# Enterprise Retail Chain Store Site Selection, Geo-Spatial LBS `Huff` Gravity Model, Cannibalization Forecasting & Commercial Lease Review Agent Architecture Selection Guide: Broker Intuition + 1km Euclidean Rings vs. Prompt-Only LLM vs. Tabular Census Regression vs. `Uber H3` Isochrone + `Huff` Gravity Cannibalization Solver + Street-View Storefront `VLM` + Lease Clause `AST` Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-retail-site-selection-geospatial-huff-lease-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, VPs of Retail Store Development & Real Estate, Location Intelligence Teams, and Commercial Legal Operations.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_retail_site_selection_geospatial_huff_lease_agent_selection/results.json`](../../experiments/enterprise_retail_site_selection_geospatial_huff_lease_agent_selection/results.json).

![Enterprise Retail Site Selection, Geo-Spatial Huff Gravity & Lease Review Agent Summary](../../assets/svg/enterprise-retail-site-selection-geospatial-huff-lease-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

For retail, QSR/coffee, convenience, pharmacy, and EV supercharging chains operating hundreds or thousands of locations, **every mislocated store destroys `$420,000` in non-recoverable build-out CapEx plus 3–5 years of dead commercial lease liabilities**. Traditional real-estate brokers using manual foot-traffic counts and **static 1-kilometer Euclidean radius circles** (Tier 1) ignore physical urban barriers (railways, rivers, divided highways), resulting in a **`34.5%` Year-1 revenue Mean Absolute Percentage Error (`MAPE`)**. Prompting a general-purpose LLM with a street address to predict sales and summarize an 85-page mall lease (Tier 2) yields **`41.2%` revenue MAPE** and misses **`11.4%`** of predatory lease riders. Even conventional tabular census regressions (Tier 3) predict gross single-store revenue in isolation—missing **`17.5%` sister-store sales cannibalization** when a new infill store steals **`38%` (`$684K/yr`)** of its revenue directly from a flagship sister store 850 meters away!

Evaluated across a 1,500-store retail chain underwriting **350 candidate locations per month and opening 240 new stores per year (`$420,000` average CapEx/store)**, the four architectural tiers compare as follows:

| Architectural Tier | Core Stack Combination | Year-1 Store Revenue `MAPE` | Sister-Store Cannibalization Error | Lease Trap Escape Rate | Site Underwriting Cycle | Monthly Underwriting & Bad-CapEx Loss | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Broker Intuition + 1km Euclidean Circle** | Manual foot-traffic tally + 1km straight-line census buffer + manual lease review | `34.5%` ✕ | `28.0%` ✕ | `18.5%` ✕ | `21.0 Days` | `$410,000 / mo` ✕ | ⛔ Blind to rivers/railways & cannibalization |
| **🟨 Tier 2: Prompt-Only General LLM Recommender** | Feeding street address & rent into an LLM to guess sales and summarize leases | `41.2%` ✕ | `35.0%` ✕ | `11.4%` ✕ | `1.5 Days` | `$495,000 / mo` ✕ | ⛔ Hallucinates traffic from street names |
| **🟦 Tier 3: Generic Tabular Census Regression** | ZIP/grid GBDT without travel-time isochrones, Huff transfer, or storefront VLM | `19.8%` | `17.5%` ✕ | `8.2%` ✕ | `5.0 Days` | `$215,000 / mo` ✕ | ⚠️ Approves stores that cannibalize flagships |
| **🟩 Tier 4: `H3` Isochrone + `Huff` Gravity + Storefront `VLM` + Lease `AST` Agent** | **Uber `H3 Res-9` LBS Grid + `OSRM` Travel-Time Isochrones + Calibrated `Huff Gravity` Cannibalization + Street-View `VLM` + Lease Clause `AST` Gate** | **`6.4%` 🏆** *(-28.1 pts)* | **`2.8%` 🏆** *(-25.2 pts)* | **`0.0%` 🏆** *(100% recall)* | **`0.25 Days (6h)` 🏆** | **`$38,000 / mo` 🏆** *(-90.7%)* | 🌟 **Recommended Architecture** |

> 💡 **Golden Rule for Retail Geo-Spatial & Real-Estate AI Architects**: **"Never draw Euclidean circles on a map and never approve a store on gross revenue alone—compute walk-time isochrones over Uber `H3` hexagons to penalize urban barriers, solve a calibrated `Huff Gravity` equation to deduct sister-store sales cannibalization, score micro-visibility via Street-View `VLM`, and enforce a deterministic Lease Clause `AST` Gate on CAM caps and co-tenancy."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Real-World Retail Expansion Pain Points)

### Q1: Why do 1km Euclidean circular buffers and standard demographic regressions fail so badly in dense urban markets?
- **First-Principles Reality**:
  1. **The 400-Meter Railway / River Barrier Illusion**: A candidate storefront sits 400 meters straight-line distance from a 25,000-worker office complex, but a fenced railway corridor forces pedestrians onto an **18-minute detour**. Tier 1's 1km Euclidean circle counts all 25,000 workers as prime customers, over-forecasting Year-1 sales by **`+68%`**.
  2. **Micro-Visibility & Commuter Side-of-Street Asymmetry**: Two storefronts on opposite sides of a 6-lane divided arterial road share the exact same ZIP code and macro census grid, yet recessed arcade pillars and median barriers reduce capture rate on the North side by **`32%`**.
  3. **Tier 4 Solution (`Uber H3 Res-9` + `OSRM/Valhalla` Walk-Time Isochrones + Street-View `VLM`)**: Tier 4 indexes mobility LBS telemetry into `Uber H3 Resolution-9` hexagons (~174m edge), computes true pedestrian travel time $t_{\text{walk}}$ via `OSRM/Valhalla` routing graphs (discounting across-barrier cells by **`94%`** via $\exp(-\beta \cdot t_{\text{walk}})$), and runs a multimodal **Street-View `VLM`** to extract storefront frontage width, pillar occlusion, and commuter ingress side into the spatial `XGBoost` ensemble.

### Q2: How does Tier 4 prevent an infill store from cannibalizing a flagship sister store 850 meters away?
- **Gross Store Sales $\neq$ Net Chain Incremental Revenue**:
  - In Scenario R2, candidate Store B is projected by an independent regression (Tier 3) to generate **`$1.38M`** in Year-1 sales. However, **`$684K` (`49.6%` of Store B's sales, or `38.0%` of Flagship Store A's `$1.80M` revenue)** is diverted directly from existing customers of Flagship Store A located 850 meters away!
  - **Calibrated `Huff Gravity` Spatial Interaction Solver**: Tier 4 computes utility-weighted capture probabilities across every origin hexagon $i$ and competing/sister store $j$:
    $$P_{ij} = \frac{A_j^\alpha \cdot T_{ij}^{-\beta}}{\sum_{k \in \text{Sister} \cup \text{Competitors}} A_k^\alpha \cdot T_{ik}^{-\beta}}$$
    reducing sister-store cannibalization error from `28.0%` to **`2.8%`** and ensuring the Investment Committee approves only sites with positive **Net Chain Incremental NPV**.

### Q3: How does the Commercial Lease Clause `AST` Gate prevent hidden real-estate lease traps?
- **Precedent-Normalized Lease Clause `AST`**:
  - Landlords frequently leave standard protections in Article 7 of a mall lease while overriding them in **Exhibit C or Rider B**—such as deleting the **5% annual cap on controllable Common Area Maintenance (`CAM`) fees**, weakening **Anchor Co-Tenancy remedies** (requiring 24 months of anchor vacancy before rent relief kicks in), or omitting **direct-competitor exclusivity**.
  - Tier 4 parses base articles and riders into a unified precedent `AST`, blocking execution (`0.0%` escape rate) whenever mandatory tenant protections are missing or overridden.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + H3/Huff/Lease Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Retail Chain Site Selection & Lease Underwriting<br/>(1,500 Existing Stores + 350 Candidate Sites/Mo + $420K CapEx/Store)"] --> B{"Uses 1km Euclidean circular buffers<br/>or prompt-only LLM address guesses?"}
    B -- "Yes (Euclidean rings or LLM guesses)" --> C["⛔ High CapEx Write-Off Risk (Tier 1 / Tier 2)!<br/>34.5%–41.2% revenue MAPE; ignores railways/rivers & lease traps"]
    B -- "Uses spatial demographic features" --> D{"Models sister-store sales cannibalization (Huff Gravity),<br/>street-view visibility & lease CAM/Co-Tenancy riders?"}
    D -- "No (Independent store regression)" --> E["⚠️ Cannibalization Blind Spot (Tier 3)<br/>17.5% cannibalization error; new stores steal flagship sales"]
    D -- "Yes (H3 Isochrone + Huff + VLM + Lease AST)" --> F["🟩 Tier 4: H3 Walk-Time Isochrone + Huff Gravity + Storefront VLM + Lease AST 🏆<br/>(6.4% Revenue MAPE | 2.8% Cannibalization Err | 0.0% Lease Trap Escape)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `Uber H3` Isochrone + `Huff Gravity` + Storefront `VLM` + Lease `AST` Topology

```mermaid
flowchart LR
    subgraph Spatial["1. H3 Isochrone & Micro-Visibility Plane"]
        LBS["LBS Footfall + POI Graph<br/>+ OSRM Pedestrian Graph"] --> H3["🔷 Uber H3 Res-9 Hexagonal Grid<br/>5/10/15-min Walk Isochrones<br/>Rail/River/Median Barrier Decay"]
        Street["Street-View Storefront Photos"] --> VLM["👁️ Storefront Visibility VLM<br/>Frontage Width / Pillar Occlusion / Ingress"]
    end

    subgraph Gravity["2. Spatial Cannibalization & Net NPV Engine"]
        H3 --> Huff["⚖️ Calibrated Huff Gravity + Spatial XGBoost<br/>Separates Gross Store Sales from<br/>Sister-Store Cannibalization ($684K)"]
        VLM --> Huff
    end

    subgraph Lease["3. Commercial Lease Clause AST Gate"]
        PDF["85-Page Mall Lease + Riders"] --> AST["🛡️ Precedent-Normalized Clause AST<br/>5% Controllable CAM Cap / Co-Tenancy<br/>& Direct-Competitor Exclusivity"]
    end

    Huff --> Gate["📋 6-Hour Investment Committee Packet<br/>(Net Chain NPV + Redlined Lease)"]
    AST --> Gate
```

### 3.3 End-to-End Sequence: Underwriting Candidate Site `#SH-084` in 6 Hours

```mermaid
sequenceDiagram
    autonumber
    participant Broker as Real-Estate Manager (Site #SH-084)
    participant Iso as H3 + OSRM Walk Isochrone Engine
    participant Huff as Storefront VLM + Huff Gravity Solver
    participant Lease as Commercial Lease Clause AST Gate
    participant IC as Store Investment Committee (IC)

    Broker->>Iso: Submit candidate #SH-084 (400m from 25K-worker tech park, 850m from Flagship A) + 85-page lease
    Iso->>Iso: Compute OSRM walk graph: fenced rail line forces 18-min detour -> apply 94% decay to across-track H3 cells
    Iso->>Huff: Pass barrier-adjusted H3 demand + Street-View pillar occlusion score
    Huff->>Huff: Solve Huff Gravity: $1.38M gross sales, but $684K (38%) cannibalized from Flagship Store A (Net NPV < 0)
    Lease->>Lease: Parse Lease + Exhibit C: detect removed 5% controllable CAM cap & missing competitor exclusivity
    Lease-->>IC: 🚨 Reject site in 6h: Negative Net Chain NPV (Confirm) + Uncapped CAM (Send to a human), saving $420K CapEx!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: 1km Euclidean Circle Excel | 🟨 Tier 2: Prompt-Only LLM | 🟦 Tier 3: Tabular Census Regression | 🟩 Tier 4: `H3` + `Huff` + `VLM` + Lease `AST` |
|---|---|---|---|---|
| **Year-1 Store Revenue Forecast `MAPE`** | `34.5%` ✕ (blind to rail/river barriers) | `41.2%` ✕ (street-name hallucination) | `19.8%` (no micro-visibility) | 🏆 **`6.4%` (`H3` walk isochrone + storefront `VLM`)** |
| **Sister-Store Cannibalization Error** | `28.0%` ✕ (manual guesswork) | `35.0%` ✕ (zero spatial math) | `17.5%` ✕ (predicts gross sales only) | 🏆 **`2.8%` (calibrated `Huff Gravity` transfer)** |
| **Commercial Lease Trap Escape Rate** | `18.5%` ✕ (misses exhibit overrides) | `11.4%` ✕ (long-context loss) | `8.2%` ✕ (naive chunk retrieval) | 🏆 **`0.0%` (precedent-merged Clause `AST` Gate)** |
| **Site Underwriting Cycle Time** | `21.0 Days` | `1.5 Days` | `5.0 Days` | 🏆 **`0.25 Days (6 hours end-to-end)`** |
| **Monthly Underwriting & Bad-CapEx Loss** | `$410,000 / mo` | `$495,000 / mo` | `$215,000 / mo` | **`$38,000 / mo` 🏆 (-90.7%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Retail Site Selection & Lease Agent Sizer](../reports/enterprise-retail-site-selection-geospatial-huff-lease-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-retail-site-selection-geospatial-huff-lease-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_retail_site_selection_geospatial_huff_lease_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_retail_site_selection_geospatial_huff_lease_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Replace Euclidean Circular Buffers with `Uber H3 Res-9` Hexagons and `OSRM/Valhalla` Walk/Drive Isochrones**: Account for rivers, railways, highway medians, and pedestrian bridge detours.
- [ ] **Mandate Net Chain Incremental NPV via Calibrated `Huff Gravity` Cannibalization Modeling**: Explicitly deduct sales transferred from existing sister stores within the shared trade area.
- [ ] **Score Micro-Location Visibility with a Street-View `VLM`**: Extract storefront width, arcade pillar occlusion, elevation steps, and commuter turn-side ingress.
- [ ] **Enforce a Precedent-Normalized Commercial Lease Clause `AST` Gate Before Signing**: Block uncapped controllable `CAM` charges, weak anchor `Co-Tenancy` remedies, and missing competitor exclusivity clauses.
- [ ] **Continuously Recalibrate Distance-Decay Exponent $\beta$ Against Post-Opening POS Actuals**: Stratify parameters by store format (mall vs. street-front vs. transit hub).
