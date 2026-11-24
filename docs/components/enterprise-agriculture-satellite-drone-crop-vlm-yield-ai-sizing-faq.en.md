# Enterprise Precision Agriculture & Food Supply AI Architecture: Satellite NDVI Funnel, 14ms Edge Spot-Spray VLM & EPA Dosage Gate Sizing Guide

[![Back to Catalog](https://img.shields.io/badge/Back_to-Catalog-blue)](../full-catalog-en.md)
[![中文版](https://img.shields.io/badge/中文-简体中文版-blue)](./enterprise-agriculture-satellite-drone-crop-vlm-yield-ai-sizing-faq.md)
[![Interactive H5 Lab](https://img.shields.io/badge/H5-Interactive_Study_Lab-10b981)](../h5/agriculture-satellite-drone-crop-vlm-yield-sizer.html)
[![7-Card Report](https://img.shields.io/badge/7--Card-Benchmark_Deck-purple)](../reports/enterprise-agriculture-satellite-drone-crop-vlm-yield-ai-sizing-cards.html)
[![Interactive Sizer](https://img.shields.io/badge/Sizer-Capacity_Calculator-orange)](../reports/enterprise-agriculture-satellite-drone-crop-vlm-yield-ai-sizing-sizer.html)

![Precision Agriculture Satellite NDVI Funnel, Drone Edge VLM & EPA Gate Overview](../../assets/svg/enterprise-agriculture-satellite-drone-crop-vlm-yield-ai-sizing-faq-summary.svg)

---

## 1. Executive Summary (3-Minute Read)

> **Bottom Line for Architects**: Scaling agricultural AI across millions of hectares fails when architects **(1) upload all raw 4K drone/satellite tiles to cloud VLMs (`$310,000/mo`, wasting `92%` of GPU cycles on healthy green canopy); (2) rely on rural cellular links for real-time weed spraying at `6 m/s` tractor speeds; or (3) allow generative LLMs to hallucinate chemical pesticide dilution ratios (`6.8%` violations of EPA/FAO pre-harvest intervals and max dosage limits)**. The production reference architecture combines a **Stage 1 `Sentinel-2` Multi-Spectral `NDVI/NDWI` Anomaly Funnel (filtering out `92%` healthy acreage) + Stage 2 On-Tractor/Drone `Jetson Orin Nano` `14 ms` Offline Spot-Spray Segmentation (`-68.4%` herbicide usage) + Stage 3 Cloud `8× L40S + 4× H200` Leaf-Crop VLM Diagnosis with a Deterministic `EPA/FAO` Agronomic SQL Gate (`0.0%` dosage/PHI violations)**—slashing monthly cloud compute from `$310K` to **`$24,800/mo` (`-92.0%`)** and achieving **`2.8%` MAPE** on regional commodity yield forecasts.

| Subsystem | Naive Cloud-Heavy Anti-Pattern | Production Space-to-Field Reference Stack | Quantified Impact |
| :--- | :--- | :--- | :--- |
| **1. 50M-Hectare Canopy Screening** | Upload all 4K drone/satellite tiles to cloud 72B VLM (`$310K/mo`, `92%` healthy canopy) | **`Sentinel-2` 10m multi-spectral `NDVI` delta isolates `8.0%` stressed zones $\to$ targeted leaf crops** | **`-92.0%` cloud VLM cost**, **`96.8%` disease F1** |
| **2. Real-Time Weed Spot-Spraying** | Cloud inference over rural 4G (`>800 ms` latency misses weeds at `6 m/s`; falls back to blanket spray) | **On-tractor `Jetson Orin Nano` (`14 ms` INT8 TensorRT) triggers PWM nozzles offline** | **`-68.4%` chemical herbicide** (`$42/ha/yr` saved) |
| **3. Agronomic Prescription Safety** | Free-form LLM chemical dosage (`6.8%` EPA/FAO label & pre-harvest interval violations) | **LLM maps to `EPPO` pest code $\to$ Deterministic `EPA/FAO SQL Gate` + wind/rain interlock** | **`0.0%` dosage or PHI violations** |
| **4. Commodity Yield Forecasting** | Prompting text LLMs with weather summaries (`9.5%` MAPE) | **`GraphCast` weather + `Chronos` soil moisture + `NDVI` ensemble (`2.8%` MAPE)** | **`2.8%` regional yield MAPE** |

---

## 2. Core Architectural Q&A

### Q1: How does the 3-tier satellite-to-tractor funnel cut cloud VLM costs by `92.0%`?
- Free 10m multi-spectral `Sentinel-2` imagery computes $\text{NDVI} = (\text{NIR} - \text{Red})/(\text{NIR} + \text{Red})$ across 50M hectares on CPU/PostGIS to flag the **`8.0%`** of parcels showing early canopy stress. Drones and tractors inspect only those stressed zones and upload only `512×512` symptomatic leaf crops to the cloud **`8× L40S + 4× H200` (`$24,800/mo`)** VLM cluster.

### Q2: How do we guarantee `0.0%` chemical dosage or Pre-Harvest Interval (`PHI`) violations?
- The LLM never generates numerical chemical quantities. It outputs a verified `EPPO` pathogen code and `BBCH` crop stage; a **deterministic SQL lookup against official EPA/FAO pesticide registration tables** computes exact `g a.i./ha`, water dilution, and checks that `Current_Date + PHI_Days <= Planned_Harvest_Date` and `Wind_Speed <= 4.0 m/s`.

---

## 3. Pre-Production Checklist

- [ ] **Offline Store-and-Forward Edge Execution**: Cache TensorRT INT8 models and parcel prescription GeoJSONs on the tractor's local NVMe for 100% offline operation.
- [ ] **Hard Pre-Harvest Interval (`PHI`) Gate**: Block any chemical prescription whose mandatory PHI window overlaps scheduled harvest dates.
- [ ] **Wind-Speed Drift Interlock**: Lock out drone spraying when local anemometer wind speed exceeds `4.0 m/s`.
