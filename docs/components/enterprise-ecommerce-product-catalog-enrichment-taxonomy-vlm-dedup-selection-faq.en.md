# Enterprise E-Commerce & Retail Product Catalog Enrichment (`2M New SKUs/Mo`), Hierarchical Taxonomy Prefix-Tree Constrained Decoding, `CLIP/SigLIP + SLM/VLM` Attribute Extraction & Multimodal Deduplication Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-ecommerce-product-catalog-enrichment-taxonomy-vlm-dedup-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, E-Commerce Marketplace / Retail `PIM & MDM` Leads, and Search & Discovery Engineers onboarding `2,000,000+` monthly SKUs across `6,500+` leaf categories.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_ecommerce_product_catalog_enrichment_taxonomy_vlm_dedup_selection/results.json`](../../experiments/enterprise_ecommerce_product_catalog_enrichment_taxonomy_vlm_dedup_selection/results.json).

![Enterprise E-Commerce Product Catalog Enrichment & Multimodal Dedup Summary](../../assets/svg/enterprise-ecommerce-product-catalog-enrichment-taxonomy-vlm-dedup-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When global e-commerce marketplaces, retail `PIM/MDM` hubs, and B2B industrial supply (`MRO`) platforms onboard **`2,000,000` new or updated SKUs per month** across **`6,500` leaf taxonomy categories** and **`40,000` canonical facet enum values** (`4` images/SKU), three common catalog-enrichment architectures fail in production:
1. **Trap #1: Unconstrained Prompting of a Frontier Vision LLM (`GPT-4o VLM`) Across All `8,000,000` Monthly Product Images (`Tier 1`, `24.5%` Non-Canonical Facet Drift — Only `75.5%` Dictionary Compliance, `16.8%` Hallucinated Unseen Specs, `$295,000/month` VLM Token Bill)**:
   - **The Broken Sidebar Search Filter Disaster**: When an unconstrained VLM outputs `"Navy"`, `"Midnight Blue"`, or `"304 SS"` instead of the exact search-index enum `"Blue"` or `"Stainless_Steel_304"`, **490,000 SKUs per month become completely invisible when shoppers filter by sidebar facets**!
2. **Trap #2: Text-Only Title Classification & Regex Extraction (`Tier 2`, `66.0%` Attribute Recall — Misses `34.0%` of Specs Printed Only on Packaging Labels or Spec Plates, `$168,000/month` Manual Merchandising TCO)**.
3. **Trap #3: Stuffing All `6,500` Taxonomy Categories & Attribute Rules into an `18,000-Token` Monolithic Prompt (`Tier 3`, `84.5%` Facet Compliance, `$220,000/month` Prompt Token Bloat)**.

Evaluated across **2,000,000 monthly SKUs (`6,500` leaf categories, `4` images/SKU)**, the four catalog architectures compare as follows:

| Catalog Enrichment Tier | Taxonomy Routing, Enum Masking & Vision Stack | Facet Dictionary Compliance | Attribute Recall | Cross-Seller Duplicate Catch | 2M-SKU Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Unconstrained Frontier `VLM` on All SKUs** | Calls Frontier VLM on all 8M images; free-text attributes | `75.5%` ✕ *(24.5% broken facets!)* | `91.0%` *(16.8% guessed specs)* | `68.0%` ✕ | `$295,000 / mo` ✕ *(massive VLM bill)* | ⛔ Free-text attribute strings break e-commerce faceted search |
| **🟨 Tier 2: Text-Only Title Classifier (`No Vision/OCR`)** | Reads seller title only; ignores packaging/spec photos | `88.0%` | **`66.0%` ✕** *(misses 34% specs!)* | `78.5%` | `$168,000 / mo` ✕ *(manual data entry)* | ⛔ Blind to voltage, thread pitch, and nutrition tables on labels |
| **🟦 Tier 3: Giant `18K-Token` Taxonomy Prompt Stuffing** | Stuffs all 6,500 categories into every LLM call | `84.5%` | `86.0%` | `74.0%` | `$220,000 / mo` ✕ *(18K tok/SKU bloat)* | ⛔ Severe context bloat and category confusion across 6,500 leaves |
| **🟩 Tier 4: Top-Down `L1->L2->L3` Router + Leaf `XGrammar` Enum Mask + `80% OCR+8B SLM / 20% VLM` + `SigLIP` Dedup** | **Routes to 1 leaf first -> loads leaf's `XGrammar` enum schema (`600t`) -> `PaddleOCR+8B SLM` (`80%`) + `SigLIP` SPU merger** | **`100.0%` 🏆** *(zero broken facets!)* | **`97.6%` 🏆** *(+31.6 pts vs Tier 2)* | **`99.2%` 🏆** *(catches cropped dupes)* | **`$28,000 / mo` 🏆** *(-90.5% vs Tier 1)* | 🌟 **Recommended Enterprise Product Catalog Enrichment Standard** |

> 💡 **Golden Rules for Enterprise Product Catalog Enrichment**:
> 1. **Two-Step Top-Down `L1 -> L2 -> L3` Routing + Leaf-Specific `XGrammar` Enum Masking**: **"Never stuff 6,500 categories into one prompt, and never allow free-text facet values!"** First route the SKU (`Title + SigLIP Image Embedding`) down the `L1 (25) -> L2 (300) -> L3 (6,500)` taxonomy tree. Once the exact leaf category is locked (e.g., `Fasteners > Hex Bolts`), dynamically load **only that leaf's 12 required attributes and canonical enum values (`~600 tokens`)** into a pre-compiled `XGrammar` schema—guaranteeing **`100.0%` facet dictionary compliance**!
> 2. **`80% PaddleOCR + Fine-Tuned 8B SLM` / `20% VLM` Split + `"NULL-if-Unseen"` Grounding**: `80%` of missing attributes (nutrition panels, MRO spec plates, care labels, voltage ratings) are printed text on packaging photos! Extract them with fast `PaddleOCR + 8B SLM` and reserve full `VLM` inference for the `20%` visual style/apparel attributes. Require an explicit `evidence_span` or `"UNSPECIFIED_NULL"` so models never hallucinate unseen specs (`0.3%` hallucination rate vs. `16.8%` in Tier 1)!
> 3. **Foreground-Segmented `SigLIP` Embeddings + Canonical Attribute Tuple Deduplication (`99.2%` Catch)**: Strip seller watermarks/borders via foreground segmentation, combine `SigLIP` visual similarity with the extracted canonical attribute tuple (`Brand + MPN + Material + Dimensions`), and merge duplicate seller listings into a single canonical `SPU`!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Fixing Broken Search Facets, Stopping Unseen Attribute Hallucinations & Cross-Seller Deduplication Pain Points)

### Q1: Why does `24.5%` of unconstrained LLM attribute extraction (`Tier 1`) fail to populate e-commerce sidebar search filters—and how does Leaf-Scoped `XGrammar` achieve `100.0%`?
- **Controlled Facet Vocabularies vs. Open-Ended Generative Synonyms**:
  - E-commerce search engines (`Elasticsearch / OpenSearch / Vespa`) index faceted navigation against strict controlled vocabularies (`material_id: "STAINLESS_STEEL_304"`). Without logit masking, LLMs emit synonyms (`"304 Stainless"`, `"SUS304"`, `"A2-70"`).
  - By routing to the leaf category first and binding that leaf's exact `Literal[...]` enum schema in `XGrammar`, every extracted attribute maps `100.0%` to a valid facet ID!

### Q2: How do we stop the model from guessing critical safety or technical specs (such as `"Allergen-Free"` or `"220V"`) when the seller's text and photos don't actually state them?
- **Evidence-First Schema Ordering + Mandatory `"UNSPECIFIED_NULL"` Escape Hatch**:
  - In Tier 4, every attribute field is paired with a preceding `evidence_quote_or_bbox` field. If neither the seller description nor the `PaddleOCR` / visual inspection contains explicit evidence, the grammar transitions to `"UNSPECIFIED_NULL"` and flags the seller portal to supply the missing spec (`0.3%` hallucination rate)!

### Q3: How do we catch `99.2%` of duplicate SKUs when 50 marketplace sellers upload the exact same product with reworded titles and cropped/watermarked photos?
- **3-Signal Multimodal Entity Resolution (`GTIN/MPN` + Foreground `SigLIP` + Canonical Attribute Hash)**:
  - Cropping an image or adding a promotional border defeats simple MD5/SHA image hashes, and rewording titles defeats BM25.
  - Tier 4 segments the foreground product object (`U2Net / SAM-Nano`), queries `SigLIP` embeddings within the same leaf category (`cosine >= 0.92`), and verifies exact match on the extracted canonical attribute tuple (`Brand + Model + Size + Spec`), catching **`99.2%`** of cross-seller duplicates!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + `L1->L3` Taxonomy, Leaf `XGrammar` & `SigLIP` Dedup Topology + Sequence)

### 3.1 Enterprise E-Commerce Product Catalog Enrichment Decision Tree

```mermaid
flowchart TD
    A["Enterprise Product Catalog Enrichment Selection<br/>(2M New SKUs/Mo | 6,500 Leaf Categories | 40K Attribute Enums | 8M Images)"] --> B{"How are SKUs classified, extracted, and deduplicated?"}
    B -- "Unconstrained Frontier VLM on all 2M SKUs (Tier 1)" --> C["⛔ 24.5% Non-Canonical Facet Drift & $295K/mo VLM Bill!<br/>490K SKUs/mo disappear from sidebar filters + 16.8% guessed specs"]
    B -- "Text-only title classifier ignoring product photos (Tier 2)" --> D["⛔ Misses 34.0% of Attributes Printed on Packaging/Labels!<br/>Only 66.0% attribute recall ($168K/mo manual rework)"]
    B -- "Giant 18K-token prompt stuffing all 6,500 categories (Tier 3)" --> E["⛔ $220,000/mo Token Bloat & 15.5% Facet Drift!<br/>Long-context confusion across 6,500 categories"]
    B -- "L1->L3 Router + Leaf XGrammar Mask + 80/20 OCR-SLM/VLM + SigLIP Dedup (Tier 4)" --> F["🟩 Tier 4: Top-Down Taxonomy + Leaf XGrammar + 80/20 OCR/VLM + SigLIP 🏆<br/>(100.0% Facet Compliance | 97.6% Recall | 99.2% Dedup | -90.5% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Top-Down `L1->L3` Router, Leaf-Scoped `XGrammar` Enum Mask & Multimodal `SigLIP` Deduplication Architecture

```mermaid
flowchart LR
    subgraph Route["1. Visual/OCR Pre-Process & L1->L3 Router"]
        SKU["📦 Inbound SKU<br/>(Title + 4 Images)"] --> Pre["⚡ Foreground SigLIP Vector<br/>+ PaddleOCR Spec Label Text"]
        Pre --> Tax["🌲 L1 -> L2 -> L3 Classifier<br/>Locks 1 of 6,500 Leaf Categories<br/>(e.g. Fasteners > Hex Bolts)"]
    end

    subgraph Extract["2. Leaf XGrammar Mask & 80/20 Extraction"]
        Tax --> Mask["📐 Load Leaf XGrammar Schema<br/>(12 leaf attributes & enums, ~600t)"]
        Mask --> Gate{"🧭 Needs Visual Style VLM?"}
        Gate -- "80% Spec/Label SKUs" --> SLM["🚀 Fine-Tuned 8B SLM + XGrammar<br/>(Evidence quote + NULL-if-Unseen)"]
        Gate -- "20% Apparel/Visual SKUs" --> VLM["👁️ Vision VLM + XGrammar"]
    end

    subgraph Dedup["3. 100% Facet Index & SPU Merger"]
        SLM & VLM --> Merge["🔗 SigLIP Cosine + Canonical Attribute Hash<br/>Catches 99.2% cross-seller duplicates -> SPU!"]
    end
```

### 3.3 End-to-End Sequence: Onboarding an Industrial SKU from a Spec-Plate Photo & Merging Duplicate Seller Listings

```mermaid
sequenceDiagram
    autonumber
    participant Seller as Seller Upload (Title + 4 Photos)
    participant Pre as SigLIP Foreground & PaddleOCR
    participant Tax as L1->L2->L3 Taxonomy Router
    participant SLM as 8B SLM (Leaf XGrammar Mask)
    participant PIM as PIM Search Index & SPU Deduplicator

    Seller->>Pre: 1. Upload SKU: Title = `"Heavy Duty Stainless Bolt M8"`, Image #3 = Packaging Spec Label
    Pre->>Tax: 2. `PaddleOCR` extracts `"ISO 4017 | A2-70 (304) | M8 x 1.25 x 40mm"`; `SigLIP` embeds foreground bolt
    Tax->>SLM: 3. Route `L1(Industrial) -> L2(Fasteners) -> L3(Hex_Bolts)` & bind pre-compiled `HexBolt` `XGrammar` schema
    SLM->>SLM: 4. Map `"A2-70 (304)"` to canonical enum `"STAINLESS_STEEL_304"`, `pitch="1.25_MM"`, `finish="UNSPECIFIED_NULL"` (`100.0%` facet match, `0` guessed fields!)
    SLM->>PIM: 5. Match `SigLIP` vector (`0.96` sim) + canonical tuple `[ISO4017, 304, M8x1.25x40]` -> merge into existing master `SPU-99102` (`99.2%` dedup)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Raw Frontier `VLM` | 🟨 Tier 2: Text-Only Title | 🟦 Tier 3: `18K` Giant Prompt | 🟩 Tier 4: `L1->L3` + Leaf `XGrammar` + `80/20` OCR/VLM |
|---|---|---|---|---|
| **Facet Dictionary Compliance** | `75.5%` ✕ (24.5% synonym drift) | `88.0%` | `84.5%` | 🏆 **`100.0%` (leaf-scoped `XGrammar` enum bitmask)** |
| **Attribute Extraction Recall** | `91.0%` | `66.0%` ✕ (misses 34% label specs) | `86.0%` | 🏆 **`97.6%` (`PaddleOCR` on spec labels + targeted `VLM`)** |
| **Cross-Seller Duplicate Catch** | `68.0%` ✕ | `78.5%` | `74.0%` | 🏆 **`99.2%` (foreground `SigLIP` + canonical attribute tuple)** |
| **Unseen Attribute Hallucination** | `16.8%` ✕ (guesses missing specs) | `4.5%` | `9.2%` | 🏆 **`0.3%` (mandatory `evidence_span` or `UNSPECIFIED_NULL`)** |
| **2M-SKU Monthly TCO** | `$295,000 / mo` ✕ | `$168,000 / mo` ✕ | `$220,000 / mo` ✕ | **`$28,000 / mo` 🏆 (-90.5% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise E-Commerce Product Catalog Enrichment Sizer](../reports/enterprise-ecommerce-product-catalog-enrichment-taxonomy-vlm-dedup-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-ecommerce-product-catalog-enrichment-taxonomy-vlm-dedup-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_ecommerce_product_catalog_enrichment_taxonomy_vlm_dedup_selection/run.py --check
  python -m unittest tests/test_enterprise_ecommerce_product_catalog_enrichment_taxonomy_vlm_dedup_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Route `L1 -> L2 -> L3` Top-Down Before Loading Leaf-Specific `XGrammar` Schemas**: Never stuff all 6,500 categories into one prompt; bind only the target leaf category's 10–15 canonical attribute enums (`100.0%` facet compliance).
- [ ] **Split Visual Extraction `80% PaddleOCR + 8B SLM` vs. `20% Vision VLM`**: Use fast OCR + SLM for packaging text, spec plates, and nutrition tables, reserving VLMs for visual fashion/style attributes (`-90.5%` TCO).
- [ ] **Enforce `"NULL-if-Unseen"` Evidence Grounding on Every Attribute**: Require an explicit `evidence_span` from the title/OCR/image before emitting a value; otherwise emit `UNSPECIFIED_NULL` and prompt the seller (`0.3%` hallucination rate).
- [ ] **Normalize Measurement Units Deterministically (`"0.5kg / 500g" -> "500_G"`)**: Prevent unit variations from fragmenting search facet buckets.
- [ ] **Deduplicate Cross-Seller Listings with Foreground-Segmented `SigLIP` Vectors + Canonical Attribute Fingerprints**: Catch `99.2%` of duplicate listings despite cropped, rotated, or watermarked seller images.
