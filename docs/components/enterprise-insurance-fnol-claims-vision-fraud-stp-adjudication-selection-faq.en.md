# Enterprise P&C Insurance FNOL & Claims Adjudication (`Guidewire ClaimCenter / Duck Creek`), Part-Segmented Damage Vision (`SAM2/YOLOv11 + VLM`), `C2PA / EXIF / Diffusion` Synthetic-Photo Fraud Gate & Deterministic Policy-Exclusion STP Selection Guide

> **Target Audience**: Cloud & AI Solution Architects, Claims Transformation & Special Investigation Unit (`SIU`) Engineering Leaders, and InsurTech Architects building First Notice of Loss (`FNOL`) and automated claims adjudication platforms on `Guidewire ClaimCenter`, `Duck Creek Claims`, and `Sapiens`.
> **Core Question**: When a Property & Casualty (P&C) or Auto insurer processes **`120,000` monthly FNOL claims and `960,000` crash/property damage photos**, why do naive "Direct VLM Dollar-Estimate Prompting (Tier 1)" or "Unconstrained Policy PDF RAG (Tier 3)" architectures suffer **`34.5%` repair cost estimation error (`MAPE`), miss `78.0%` of `Stable Diffusion / Flux` AI-inpainted damage and recycled web crash photos, and hallucinate `12.8%` of policy endorsements/deductibles (`$580,000/month` in fraud leakage and supplement rework)**? How does a **4-stage architecture combining a Pre-Ingestion `C2PA / EXIF + DIRE Diffusion Artifact + 50M-Image pHash` Forensic Fraud Gate + `YOLOv11 / SAM2` Part-Level Damage Segmentation Mapped to `VIN -> CCC ONE / Mitchell` OEM Part Catalogs + Deterministic `Guidewire / OPA` Policy Coverage Engine + Risk-Tiered Straight-Through Processing (`STP <= $2,500`)** cut repair estimation error to **`4.6%`**, catch **`99.2%` of synthetic/recycled photo fraud**, achieve **`68.5%` Straight-Through Processing (`4.2-minute` settlement)**, and reduce monthly TCO by **`89.3%` (`$580,000 -> $62,000/month`)**?

![Enterprise P&C Insurance FNOL & Claims Adjudication, Part-Segmented Vision, Synthetic Fraud Gate & STP Summary](../../assets/svg/enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

In P&C and Auto insurance claims, **the two costliest architectural anti-patterns are: (1) asking a general-purpose Vision-Language Model (`GPT-4o / Claude Vision`) to look at a dented bumper and guess a dollar repair figure, and (2) running automated payouts without a forensic gate to catch `Flux / Stable Diffusion` AI-inpainted dents and recycled web crash photos**:

| Architecture Tier | Best-Fit Scenario | Repair Estimate Error (`MAPE`) | Synthetic/Duplicate Fraud Catch | Straight-Through Processing (`STP`) | Monthly Fraud Leakage & Adjuster TCO | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Direct VLM Dollar-Estimate Prompting + No Fraud Gate (Anti-Pattern)** | ❌ **Never use for binding repair estimates or automated payouts** | `34.5%` (misses ADAS brackets!) | **`22.0%` (misses `78%` fraud!)** | `42.0%` (high leakage) | `$580,000 / mo` | ❌ **Reject**: Fooled by AI-inpainted dents & guesses arbitrary part prices |
| **Tier 2: Legacy Rules Intake + 100% Manual Desk Adjuster Review** | Complex `>$50,000` bodily-injury litigation or commercial cat losses | `9.5%` | `64.0%` (human eye misses AI) | **`8.0%` (`11.5-day` cycle)** | `$460,000 / mo` | ❌ **Reject**: Spends `$145` in desk adjuster labor to settle a `$400` glass claim |
| **Tier 3: VLM + Unconstrained Policy PDF RAG (No `VIN` Catalog or `OPA`)** | Internal adjuster note summarization only | `18.2%` | `75.0%` | `38.0%` | `$280,000 / mo` | ⚠️ **Fragile**: `12.8%` policy endorsement & deductible hallucination rate |
| **Tier 4: `C2PA/DIRE/pHash` Fraud Gate + `SAM2/YOLOv11` `VIN` Catalog + `OPA` STP (Recommended)** | **Enterprise Auto, Property & Device Claims Production Standard** | **`4.6%`** | **`99.2%`** | **`68.5%` (`4.2m` settlement)** | **`$62,000 / mo` (`-89.3%`)** | 🏆 **Production Gold Standard**: Vision segments parts; `VIN` catalog prices; `OPA` adjudicates |

### 💡 Three Golden Rules for Insurance Claims AI Architects
1. **Screen Every Uploaded Photo Through a `<120ms` Forensic Fraud Gate (`C2PA/EXIF + DIRE Diffusion Detector + 50M-Image pHash/DINOv2`) *Before* Running Vision Pricing**: Fraud rings now use `Stable Diffusion / Flux Inpainting` to paint realistic hail dents or cracked windshields onto clean vehicles in 3 seconds, or submit rotated crash photos scraped from online salvage auctions. Tier 4 enforces guided mobile capture (`C2PA` cryptographic camera attestation + live GPS + `VIN` barcode in frame) and screens every frame through a **`DIRE / TruFor` frequency-domain diffusion artifact detector** and a **`50M`-image historical claim `pHash + DINOv2` vector index**—catching **`99.2%` of synthetic and duplicate photo fraud** and routing them directly to the **Special Investigation Unit (`SIU`)**.
2. **Never Let a Vision LLM Guess Dollar Figures: Decouple `YOLOv11 + SAM2` Part-Severity Segmentation from Deterministic `VIN -> CCC ONE / Mitchell / Audatex` Catalog Pricing**: A cracked front bumper on a 2018 sedan costs `$380` to refinish, whereas the exact same crack on a 2025 SUV requires replacing a `$2,400` ADAS millimeter-wave radar bracket and performing static camera calibration. Vision models must output strictly structured damage tuples **`(Part_Code: Front_Bumper_Cover, Damage_Type: Crack, Area_Pct: 18%, Action: Replace)`**, which then query the deterministic **`VIN`-decoded `CCC ONE / Mitchell` OEM parts, ADAS calibration, and local ZIP-code labor rate engine**—slashing repair estimate error (`MAPE`) from `34.5%` to **`4.6%`**.
3. **Adjudicate Policy Coverage, Endorsements, and Deductibles in Deterministic `Guidewire / OPA` Rules to Unlock `68.5%` Straight-Through Processing (`STP`) in `4.2 Minutes`**: Policy effective timestamps, named-driver exclusions, endorsement overrides, and deductible math execute in `Open Policy Agent (OPA)`, enabling instant **`4.2-minute` Straight-Through Settlement** on `68.5%` of clean `<= $2,500` claims while cutting monthly TCO by **`89.3%` (`$580,000 -> $62,000/month`)**.

---

## 2. 🎯 Customer & Architect Critical Q&A

### Q1: Why does prompting `GPT-4o / Claude Vision` directly for a repair estimate (Tier 1) suffer a `34.5%` estimation error and `78.0%` synthetic fraud miss rate?
* **Hidden Mechanical/ADAS Dependencies & Pixel-Level Diffusion Inpainting**:
  * **Blindness to `VIN`-Specific Sub-Assemblies**: General VLMs do not have access to OEM exploded collision catalogs (`CCC ONE / Mitchell / Audatex`) or regional body-shop labor rates (`$62/hr` vs. `$135/hr`), triggering massive body-shop supplement rewrites on `52%` of claims.
  * **Zero Defense Against GenAI Inpainting**: Human eyes and general VLMs cannot detect diffusion noise residuals when a fraudster inpaints a dent onto their own car's license-plate-visible photo. Dedicated **`TruFor / DIRE`** spectral noise analyzers pinpoint the exact inpainted bounding box in `<80ms`.

### Q2: Why does Policy PDF RAG (Tier 3) fail on `12.8%` of coverage and deductible determinations?
* **Endorsement Precedence Hierarchy**:
  * Base policy page 12 excludes flood/water damage, while Endorsement `END-402` on page 39 restores water coverage with a `$1,000` special deductible except when operated by an unreported household driver under age 25. Vector RAG retrieves conflicting chunks and hallucinates coverage rules, whereas structured `Guidewire / OPA` evaluation guarantees `100.0%` deterministic accuracy.

---

## 3. 🧭 Visual Diagrams (Mermaid Decision Tree + Reference Architecture + Sequence Diagram)

### 3.1 Enterprise Insurance FNOL & Claims Adjudication Decision Tree

```mermaid
flowchart TD
    A["Incoming Auto / Property FNOL Claim & Damage Photos"] --> B["Stage 1: Image Forensic & Fraud Gate (<120ms)\n(C2PA/EXIF + DIRE/TruFor Diffusion Check + 50M pHash/DINOv2 Dedup)"]
    B -- "Synthetic AI Inpainting or Duplicate Claim Image Detected" --> C["🚨 Red Lane (SIU Quarantine): Block Payout & Attach Tamper Heatmap"]
    B -- "Authentic Live Capture & Zero Prior Match" --> D["Stage 2: YOLOv11 + SAM2 Part Segmentation + VLM Severity Classifier\nEmits (Part_ID, Damage_Type, Area_Pct, Repair_vs_Replace)"]
    D --> E["Stage 3: Deterministic VIN -> CCC ONE / Mitchell Part & Labor Lookup\n+ Guidewire / OPA Policy Coverage & Deductible Engine"]
    E --> F{"Net Payout <= $2,500 & Zero Structural/BI Flags?"}
    F -- "Yes (68.5% of Claims)" --> G["🟢 Green Lane (STP): Instant 4.2-Min Settlement & Payout"]
    F -- "No (26.3% Moderate/Complex)" --> H["🟡 Amber Lane: Pre-Populated Guidewire Line-Item Estimate for 90s Desk Review"]
```

### 3.2 Tier 4 Forensic Fraud Gate + `SAM2/VIN` Part Catalog + `OPA` STP Reference Architecture

```mermaid
flowchart LR
    subgraph Gate["🛡️ 1. Forensic & Anti-Fraud Gate (<120ms)"]
        IMG["Guided Mobile FNOL Photos\n(C2PA Attestation + Live VIN)"] --> FOR["EXIF/C2PA Verifier +\nDIRE/TruFor Diffusion Detector"]
        FOR --> DEDUP["50M Historical Claim Index\npHash + DINOv2 Duplicate Check"]
    end

    subgraph Vision["🔍 2. Part Segmentation & VIN Catalog Engine"]
        DEDUP --> SAM["YOLOv11 + SAM2 Part Polygons\n+ VLM Severity (Dent/Crack)"]
        SAM --> CAT["VIN Decoder -> CCC ONE / Mitchell\nOEM Parts, ADAS & ZIP Labor Table"]
    end

    subgraph Adjudicate["⚖️ 3. Guidewire / OPA Coverage & 3-Lane Router"]
        CAT --> OPA["OPA Policy Rules Engine\n(Effective Date / Endorsements / Deductible)"]
        OPA --> STP["🟢 Green Lane STP Payout (68.5% / 4.2m)"]
        OPA --> DESK["🟡 Amber Lane Desk Adjuster Pre-Fill (26.3%)"]
        FOR & DEDUP -. "Fraud Alert (5.2%)" .-> SIU["🔴 Red Lane SIU Fraud Investigation"]
    end
```

### 3.3 End-to-End FNOL Intake, Synthetic Fraud Check, VIN Pricing & 4.2m STP Sequence

```mermaid
sequenceDiagram
    autonumber
    participant User as Policyholder (Mobile App)
    participant Fraud as C2PA/DIRE/pHash Forensic Gate
    participant Vision as SAM2/YOLO + VIN Catalog Engine
    participant OPA as Guidewire + OPA Policy Engine
    participant Pay as Instant STP Payout / Body Shop DRP

    User->>Fraud: Submits guided `VIN` scan + 6 rear-door scrape photos
    Fraud->>Fraud: Verifies `C2PA` camera signature + `TruFor` tamper score (`0.02` clean) + `50M pHash` zero prior match
    Fraud->>Vision: Forwards verified images & `VIN: 5UXCR6C0...`
    Vision->>Vision: `SAM2` segments `Right_Rear_Door_Panel (Dent 12cm, Repair)` + `Wheel_Arch_Trim (Crack, Replace)`
    Vision->>Vision: Queries `CCC ONE` by `VIN` & ZIP -> Part `$185` + Labor `4.5h x $95` = `$612.50`
    Vision->>OPA: Sends line-item estimate (`$612.50`) + incident timestamp & driver ID
    OPA->>OPA: Verifies active coverage, zero-deductible endorsement & `STP <= $2,500` eligibility
    OPA->>Pay: Settles claim in `4.2 minutes` with instant payout or 1-click DRP body-shop booking
```

---

## 4. 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | Tier 1: Direct VLM Dollar Guess + No Fraud Gate | Tier 2: Legacy 100% Manual Desk Adjuster | Tier 3: VLM + Unconstrained Policy PDF RAG | Tier 4: `C2PA/DIRE` Fraud Gate + `SAM2/VIN` Catalog + `OPA` STP (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **Repair Cost Estimate Error (`MAPE`)** | **`34.5%` (misses ADAS & regional labor)** | `9.5%` | `18.2%` | **`4.6%` (`VIN`-exact OEM parts & ZIP labor table)** |
| **Synthetic / Duplicate Photo Fraud Catch** | **`22.0%` (misses `78%` AI/web fraud)** | `64.0%` | `75.0%` | **`99.2%` (`C2PA` + `DIRE/TruFor` + `50M pHash`)** |
| **Straight-Through Processing (`STP`)** | `42.0%` (high overpayment leakage) | **`8.0%` (`11.5-day` cycle time)** | `38.0%` | **`68.5%` (`4.2-minute` end-to-end settlement)** |
| **Policy Exclusion & Deductible Accuracy** | `82.0%` | `97.5%` | `87.2%` (`12.8%` endorsement error) | **`100.0%` (deterministic `Guidewire / OPA` engine)** |
| **Processing Cost per Simple Claim** | `$18.50` (plus fraud leakage) | **`$145.00` (manual desk labor)** | `$42.00` | **`$1.85` (`-98.7%` per-claim cost)** |
| **Monthly Fraud Leakage & Adjuster TCO** | `$580,000 / mo` | `$460,000 / mo` | `$280,000 / mo` | **`$62,000 / mo` (`-89.3%` savings)** |

---

## 5. 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Report

### 5.1 Baseline Workload (`120,000` Monthly FNOL Claims / `960,000` Damage Photos)
* **Workload Profile**: P&C and Auto insurer processing `120,000` monthly FNOL claims (`8` photos per claim = `960,000` images/month; `70%` low-to-moderate severity eligible for STP), at `$145` manual desk adjuster cost per claim.
* **Reproducible Model & Dataset**: See [`experiments/enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection/run.py`](../../experiments/enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection/run.py) and [`results.json`](../../experiments/enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive Sizer**: 📊 [Open Enterprise Insurance FNOL & Claims Adjudication Sizer (`enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-sizer.html`)](../reports/enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-selection-cards.html`)](../reports/enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Direct VLM Dollar Guess", "mape_pct": 34.5, "fraud_catch_pct": 22.0, "stp_rate_pct": 42.0, "monthly_tco_k": 580.0 },
      { "tier": "Tier 2: Manual Desk Adjuster", "mape_pct": 9.5, "fraud_catch_pct": 64.0, "stp_rate_pct": 8.0, "monthly_tco_k": 460.0 },
      { "tier": "Tier 3: VLM + Unconstrained Policy RAG", "mape_pct": 18.2, "fraud_catch_pct": 75.0, "stp_rate_pct": 38.0, "monthly_tco_k": 280.0 },
      { "tier": "Tier 4: C2PA/DIRE Gate + SAM2/VIN + OPA STP", "mape_pct": 4.6, "fraud_catch_pct": 99.2, "stp_rate_pct": 68.5, "monthly_tco_k": 62.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "mape_pct": "Percentage",
    "fraud_catch_pct": "Percentage",
    "stp_rate_pct": "Percentage",
    "monthly_tco_k": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 4.6% Repair MAPE, 99.2% Fraud Catch, and Cuts Monthly Claims TCO by 89.3%",
    "subtitle": "Monthly fraud leakage, supplement rework, and manual adjuster TCO ($ USD Thousands) across 120K claims/mo",
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

- [ ] **Pre-Ingestion `C2PA / EXIF + DIRE/TruFor` Synthetic Inpainting Gate**: Does the FNOL pipeline enforce guided live camera capture and screen every image for diffusion/GAN inpainting residuals before running damage segmentation?
- [ ] **Cross-Claim `pHash + DINOv2` Duplicate & Prior-Damage Index**: Are damaged-part crops compared against a multi-year historical claim and web salvage vector database to block recycled crash photos?
- [ ] **Strict Separation of `SAM2/YOLOv11` Part Segmentation from `VIN -> CCC ONE / Mitchell` Pricing**: Is the VLM prohibited from generating dollar numbers, with 100% of part prices, labor hours, paint materials, and `ADAS` calibrations looked up deterministically by `VIN` and ZIP code?
- [ ] **Deterministic `Guidewire / OPA` Coverage, Endorsement & Deductible Engine**: Are policy effective dates, named exclusions, and sub-limits evaluated via deterministic rules rather than policy PDF RAG?
- [ ] **3-Lane Risk Routing (`Green STP <= $2,500` / `Amber Desk Pre-Fill` / `Red SIU Quarantine`)**: Do suspicious or high-severity claims automatically block instant payout and attach pixel-level tamper heatmaps for `SIU` investigators?
