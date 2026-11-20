# Enterprise Supply Chain Demand Forecasting & Automated Replenishment (`SAP IBP / Blue Yonder / Kinaxis`), Time-Series Foundation Models (`Chronos / TimesFM + LightGBM`) vs. Text-LLM Forecasting, Unstructured Promo Covariate Fusion & `P95` Quantile Safety-Stock Selection Guide

> **Target Audience**: Cloud & AI Solution Architects, Supply Chain Data Science Leaders, and ERP/Planning Platform Directors building demand forecasting and automated inventory replenishment systems across `SAP Integrated Business Planning (IBP)`, `Blue Yonder`, `Kinaxis RapidResponse`, and `Oracle SCM`.
> **Core Question**: When an enterprise forecasts **`500,000` active `SKU-Location` combinations every week (`2.0M` weekly forecasts, where `18.0%` are cold-start new product launches with `<8` weeks of sales history)** to drive automated purchase orders, why do naive "Prompting Text LLMs (`GPT-4o / Claude`) Directly with CSV Sales History (Tier 1)" or "Legacy Univariate `ARIMA / ETS` Extrapolation (Tier 2)" suffer **`28.4%` Weighted Absolute Percentage Error (`WAPE`), `42.0%` cold-start new-SKU failure, `9.2%` shelf stockout rates, and `$640,000/month` in lost sales, spoilage, and token bills**? How does a **4-stage architecture combining an LLM Unstructured Exogenous Covariate Featurizer + Hybrid `LightGBM` (Dense SKUs) & Zero-Shot Time-Series Foundation Models (`Chronos-Bolt / TimesFM` for Cold-Start SKUs) + Calibrated `P50/P90/P95` Conformal Quantiles + `Google OR-Tools` Newsvendor Replenishment Solver** cut overall `WAPE` to **`9.4%`**, reduce cold-start `WAPE` to **`12.8%`**, drop shelf stockouts to **`1.1%`**, and slash monthly supply chain TCO by **`89.4%` (`$640,000 -> $68,000/month`)**?

![Enterprise Supply Chain Demand Forecasting, Chronos/TimesFM + LightGBM & Safety-Stock Summary](../../assets/svg/enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

In enterprise supply chain planning, **the single worst architectural anti-pattern is pasting raw CSV sales history into a general-purpose text LLM (`GPT-4o / Claude`) and asking it to output future weekly sales numbers—BPE text tokenizers fragment continuous numbers into arbitrary subword tokens, fail on seasonality and cross-SKU cannibalization, and burn `$85,000/month` in API tokens across `500,000` SKUs**:

| Architecture Tier | Best-Fit Scenario | Overall Forecast Error (`WAPE`) | Cold-Start New-SKU `WAPE` | Shelf Stockout Rate | Monthly Stockout, Holding & Compute TCO | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Prompting Text LLMs (`GPT-4o`) Directly with Sales CSVs (Anti-Pattern)** | ❌ **Never use for quantitative time-series forecasting** | `28.4%` (severe drift!) | `42.0%` | `9.2%` | `$640,000 / mo` (`$85K` tokens!) | ❌ **Reject**: BPE tokenizers butcher numbers & cost 200x more than TSFMs |
| **Tier 2: Legacy Univariate `ARIMA / ETS` Per-SKU Extrapolation** | Stationary utility consumables with zero promos or new launches | `22.6%` | `39.0%` (no history) | `6.8%` | `$410,000 / mo` | ❌ **Reject**: Blind to promo calendars, cannibalization, and new SKUs |
| **Tier 3: Point-Forecast `LightGBM` Only (No TSFM Cold-Start or `P95` Quantiles)** | Mature SKUs with `>2` years of clean history | `15.8%` | `31.5%` (`<8`-wk starvation) | `4.8%` (lead-time tail risk) | `$260,000 / mo` | ⚠️ **Fragile**: Single-point `P50` averages stock out during lead-time spikes |
| **Tier 4: `LightGBM + Chronos-Bolt/TimesFM` + LLM Covariates + `P95` Conformal + `OR-Tools` (Recommended)** | **Enterprise Retail, CPG, Auto & Pharma Supply Chain Production Standard** | **`9.4%`** | **`12.8%`** | **`1.1%` (`98.9%` in-stock)** | **`$68,000 / mo` (`-89.4%`)** | 🏆 **Production Gold Standard**: LLMs extract features; `TSFM+GBDT` forecast; `OR-Tools` orders |

### 💡 Three Golden Rules for Supply Chain AI Architects
1. **Use Text LLMs Strictly to *Extract Exogenous Covariate Features* from Unstructured Documents (`$120/mo`), Never to Extrapolate Time-Series Numbers (`$85K/mo`)**: Marketing campaign PDFs, regional trade-promotion emails, and port-congestion bulletins are unstructured text that `ARIMA` and `LightGBM` cannot read directly. Run a lightweight LLM once per document (`$120/mo` total) to parse unstructured briefs into structured numerical covariates (`promo_discount_pct = 0.25`, `display_endcap = 1`, `supplier_lead_time_delta_days = +6`) and feed those structured columns into your time-series engine.
2. **Combine `LightGBM` (Mature Dense SKUs) with Zero-Shot Time-Series Foundation Models (`Chronos-Bolt / Google TimesFM` for `<8-Week` Cold-Start & Intermittent SKUs)**: Gradient-boosted trees (`LightGBM`) excel on mature SKUs with rich lag features (`lag_7d`, `lag_52w`), but degrade to `31.5% WAPE` on the `18%` of catalog SKUs that launched `<8` weeks ago. Pre-trained patch time-series foundation models (**`Amazon Chronos-Bolt` or `Google TimesFM`**) transfer zero-shot seasonal and launch-decay priors across categories, slashing cold-start `WAPE` from `39.0%` to **`12.8%`** and overall `WAPE` to **`9.4%`**.
3. **Never Cut Purchase Orders from Single-Point Averages (`P50`): Calibrate `P50 / P90 / P95` Conformal Quantiles and Solve Replenishment via `Google OR-Tools`**: Because stocking out on a `45%`-margin product costs `5–10x` more than holding one unit of safety stock, use **`MAPIE` Conformal Quantile Regression** to output calibrated `P50/P90/P95` demand distributions convolved with supplier lead-time variance, then run **`Google OR-Tools`** to enforce minimum order quantities (`MOQ`), pallet/container rounding, and shelf-life constraints—cutting shelf stockouts to **`1.1%`**.

---

## 2. 🎯 Customer & Architect Critical Q&A

### Q1: Why does feeding 52 weeks of historical sales in a CSV prompt to `GPT-4o / Claude` (Tier 1) fail so badly on both accuracy (`28.4% WAPE`) and cost (`$85,000/month`)?
* **BPE Subword Tokenization & Per-SKU Token Multiplication**:
  * **Tokenization Blindness**: General-purpose text tokenizers split multi-digit numbers inconsistently (`"1428"` -> `["14", "28"]` vs. `"1429"` -> `["142", "9"]`), destroying autoregressive lag math and Fourier periodicity.
  * **Cost Explosion**: Prompting a frontier LLM with `3,000` input tokens per SKU across `500,000` SKU-store combinations every week consumes `6 Billion tokens/month` (`~$85,000/mo`), whereas running `LightGBM + Chronos-Bolt` across all `500,000` series on a single `NVIDIA L40S` GPU finishes in **`18 minutes` (`$420/month`)**.

### Q2: Why does standalone `LightGBM` (Tier 3) still suffer `31.5%` cold-start WAPE and `4.8%` stockouts?
* **Missing Autoregressive Lags on New Launches & Point-Forecast Tail Blindness**:
  * On a new product in Week 2, `lag_28d` and `lag_365d` are `NaN`. Routing `<8-week` SKUs to **`Chronos-Bolt / TimesFM`** conditioned on analogous SKU embeddings cuts cold-start error to `12.8%`. Furthermore, replacing single-point `P50` predictions with calibrated **`P95` conformal intervals** protects against erratic promo spikes and supplier delays.

### Q3: How do you ensure store-level forecasts add up to regional and national totals (`Hierarchical Coherence`)?
* **Minimum Trace (`MinT`) Hierarchical Reconciliation**:
  * Independent store-level and national-level models inevitably disagree when summed. Tier 4 applies **`MinT` optimal hierarchical reconciliation** across the `Store -> DC -> Region -> National` tree before handing `P50/P90/P95` quantiles to the `OR-Tools` replenishment solver.

---

## 3. 🧭 Visual Diagrams (Mermaid Decision Tree + Reference Architecture + Sequence Diagram)

### 3.1 Enterprise Supply Chain Demand Forecasting Decision Tree

```mermaid
flowchart TD
    A["Enterprise Weekly Demand Forecasting & Replenishment (500K SKU-Locations)"] --> B{"Input Data Modality?"}
    B -- "Unstructured Promo PDFs / Port Delay Alerts / Weather Bulletins" --> C["Run Lightweight LLM Covariate Featurizer ($120/mo)\nNEVER ask a text LLM to extrapolate sales numbers!"]
    B -- "Structured POS Sales History + Inventory / Price Tables" --> D{"Does the SKU have >= 8 weeks of non-zero history?"}
    C --> D
    D -- "Yes (82% Mature Dense SKUs)" --> E["Primary Engine: LightGBM Multi-Quantile + Cross-SKU Cannibalization Features"]
    D -- "No (18% Cold-Start Launches / Intermittent Spare Parts)" --> F["Cold-Start Engine: Chronos-Bolt / TimesFM Zero-Shot TSFM + Analogous SKU Anchor"]
    E & F --> G["MAPIE Conformal Quantile Calibration + MinT Hierarchical Reconciliation (P50/P90/P95)"]
    G --> H["Google OR-Tools Newsvendor MIP Solver (Margin x Lead-Time Dist x MOQ x Pallet Rounding) -> SAP IBP"]
```

### 3.2 Tier 4 LLM Covariate Featurizer + `LightGBM + Chronos` + `OR-Tools` Architecture

```mermaid
flowchart LR
    subgraph Feat["📄 1. LLM Unstructured Covariate Layer ($120/mo)"]
        DOC["Promo Brief PDFs / Competitor\nAlerts / Port Delay Notices"] --> LLM["8B/70B Structured Covariate Extractor\nEmits promo_depth, lt_delta_days"]
    end

    subgraph Forecast["📈 2. Hybrid TSFM + GBDT & Conformal Quantile Layer (18 min)"]
        ERP["SAP / Snowflake POS Sales\n& Inventory History (Uncensored)"] --> ROUTER{"SKU Lifecycle Router\n(History >= 8 Wks?)"}
        LLM --> ROUTER
        ROUTER -- "Mature SKUs (82%)" --> GBDT["LightGBM Multi-Quantile Engine\n(Autoregressive Lags & Price Elasticity)"]
        ROUTER -- "Cold-Start / Sparse (18%)" --> TSFM["Chronos-Bolt / TimesFM\nPre-Trained Time-Series Foundation Model"]
        GBDT & TSFM --> CONF["MAPIE Conformal Calibration + MinT\nCalibrated P50 / P90 / P95 Intervals"]
    end

    subgraph Optim["📦 3. Operations Research Replenishment Solver"]
        CONF --> OR["Google OR-Tools Newsvendor MIP\n(Margin x Lead-Time Variance x MOQ x Pallet)"]
        OR --> SAP["Write Back to SAP IBP / Kinaxis\nApprove-Ready Purchase Orders (PO)"]
    end
```

### 3.3 End-to-End Promo Extraction, Hybrid Quantile Forecasting & OR-Tools PO Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Mkt as Marketing PDF & Logistics Alert
    participant LLM as LLM Covariate Featurizer
    participant Engine as LightGBM + Chronos-Bolt Engine
    participant Solver as OR-Tools Replenishment Solver
    participant SAP as SAP IBP / ERP Workbench

    Mkt->>LLM: Uploads `Summer_TikTok_Campaign.pdf` & `Port_Congestion_+5d_Notice.eml`
    LLM->>Engine: Emits structured covariates: `SKU-8821: promo_tier=S, discount=0.25, supplier_lt_add_days=5`
    Engine->>Engine: Uncensors past stockouts; runs `LightGBM` (mature) + `Chronos-Bolt` (cold-start) -> `P50=1,200, P95=1,680`
    Engine->>Solver: Streams calibrated `(P50, P90, P95)` demand quantiles for `500,000` SKU-locations
    Solver->>Solver: Solves Newsvendor MIP with `42% margin`, `MOQ=500`, `60 cases/pallet` -> Optimal PO = `1,620 cases (27 pallets)`
    Solver->>SAP: Staging PO in `SAP IBP` with human-readable driver attribution ("+28% safety stock for Tier-S promo & +5d port delay")
```

---

## 4. 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | Tier 1: Text-LLM CSV Prompting (`GPT-4o`) | Tier 2: Legacy Univariate `ARIMA / ETS` | Tier 3: Point-Only `LightGBM` (No TSFM / No Quantiles) | Tier 4: `LightGBM + Chronos-Bolt` + LLM Covariates + `P95` Conformal + `OR-Tools` (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Forecast Error (`WAPE`)** | **`28.4%` (BPE tokenization drift)** | `22.6%` | `15.8%` | **`9.4%` (Best-in-class accuracy)** |
| **Cold-Start SKU `WAPE` (`<8 Wks`)** | `42.0%` | `39.0%` | `31.5%` (missing lag features) | **`12.8%` (`Chronos-Bolt / TimesFM` zero-shot transfer)** |
| **Shelf Stockout Rate** | `9.2%` | `6.8%` | `4.8%` (mean forecast fails on spikes) | **`1.1%` (`P95` conformal quantile x lead-time variance)** |
| **Unstructured Promo / Disruption Awareness** | ⚠️ Reads text, botches math | ❌ 100% blind | ❌ Manual feature engineering | **✅ Automated LLM covariate extraction (`$120/mo`)** |
| **Batch Compute Cost (`500K` SKUs/wk)** | `$85,000 / mo` (API tokens) | `$1,200 / mo` | `$350 / mo` | **`$420 / mo` (`18 min` on 1x L40S GPU + CPU)** |
| **Monthly Stockout, Holding & Compute TCO** | `$640,000 / mo` | `$410,000 / mo` | `$260,000 / mo` | **`$68,000 / mo` (`-89.4%` savings)** |

---

## 5. 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Report

### 5.1 Baseline Workload (`500,000` Active `SKU-Location` Combinations / `2.0M` Weekly Forecasts)
* **Workload Profile**: Omnichannel retail and CPG supply chain forecasting `500,000` active `SKU-Location` pairs weekly (`18.0%` cold-start products with `<8` weeks of history), targeting a `>=98.5%` in-stock fill rate.
* **Reproducible Model & Dataset**: See [`experiments/enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection/run.py`](../../experiments/enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection/run.py) and [`results.json`](../../experiments/enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive Sizer**: 📊 [Open Enterprise Supply Chain Demand Forecasting & Safety-Stock Sizer (`enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-sizer.html`)](../reports/enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-selection-cards.html`)](../reports/enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Text-LLM CSV Prompting", "wape_pct": 28.4, "cold_start_wape_pct": 42.0, "stockout_pct": 9.2, "monthly_tco_k": 640.0 },
      { "tier": "Tier 2: Univariate ARIMA/ETS", "wape_pct": 22.6, "cold_start_wape_pct": 39.0, "stockout_pct": 6.8, "monthly_tco_k": 410.0 },
      { "tier": "Tier 3: Point-Only LightGBM", "wape_pct": 15.8, "cold_start_wape_pct": 31.5, "stockout_pct": 4.8, "monthly_tco_k": 260.0 },
      { "tier": "Tier 4: LightGBM + Chronos + P95 Conformal", "wape_pct": 9.4, "cold_start_wape_pct": 12.8, "stockout_pct": 1.1, "monthly_tco_k": 68.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "wape_pct": "Percentage",
    "cold_start_wape_pct": "Percentage",
    "stockout_pct": "Percentage",
    "monthly_tco_k": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 9.4% WAPE, 1.1% Stockout Rate, and Cuts Monthly Supply Chain TCO by 89.4%",
    "subtitle": "Monthly stockout penalties, excess working-capital holding, and compute TCO ($ USD Thousands)",
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

- [ ] **Ban Direct Text-LLM Numerical Time-Series Extrapolation**: Are text LLMs restricted strictly to extracting structured covariate columns from unstructured promo/logistics documents rather than generating numerical sales forecasts?
- [ ] **Lifecycle-Aware Routing (`LightGBM` for `>=8 Wks` + `Chronos-Bolt / TimesFM` for `<8 Wks` Cold-Start)**: Does the pipeline route cold-start and intermittent SKUs to a pre-trained time-series foundation model anchored on analogous product attributes?
- [ ] **Historical Stockout Demand Uncensoring**: Are historical zero-sales days when on-hand inventory was `0` imputed/uncensored prior to training so the model does not confuse stockouts with zero customer demand?
- [ ] **Calibrated `MAPIE Conformal Quantiles (P50/P90/P95)` & `MinT` Hierarchical Reconciliation**: Are store, DC, and national forecasts reconciled via `MinT` and paired with calibrated `P95` upper bounds for safety-stock sizing?
- [ ] **`Google OR-Tools` Replenishment Constraint Solver**: Are `MOQ`, pallet/container rounding, and shelf-life limits enforced deterministically before staging purchase orders in `SAP IBP / Kinaxis`?
