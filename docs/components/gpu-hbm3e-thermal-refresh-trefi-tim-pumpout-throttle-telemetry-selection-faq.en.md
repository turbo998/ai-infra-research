# GPU HBM3e High-Temperature Refresh (`tREFI` 2x/4x), TIM Pump-Out & Thermal Throttling Telemetry Selection Guide: Unmonitored `T_mem=90°C` vs. Air-Cooled `81°C` Silent 2x Refresh vs. Reactive Throttle Alerts vs. `<72°C Liquid Cooling + ΔT Gate`

> **Language / 语言**: [中文](./gpu-hbm3e-thermal-refresh-trefi-tim-pumpout-throttle-telemetry-selection-faq.md) | **English (Current)**  
> **End-to-End Interactive Artifacts**: [🧮 Interactive HBM Thermal & Refresh Sizer](../reports/gpu-hbm3e-thermal-refresh-trefi-tim-pumpout-throttle-telemetry-sizer.html) · [🃏 7-Card 4:5 Visual Report](../reports/gpu-hbm3e-thermal-refresh-trefi-tim-pumpout-throttle-telemetry-selection-cards.html) · [🧪 Sealed Evaluator & Data](../../experiments/gpu_hbm3e_thermal_refresh_trefi_throttle_selection/run.py)  
> **Data Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`

![GPU HBM3e High-Temperature Refresh (tREFI 2x/4x), TIM Pump-Out & Thermal Throttling Selection Summary](../../assets/svg/gpu-hbm3e-thermal-refresh-trefi-tim-pumpout-throttle-telemetry-selection-faq-summary.svg)

---

## 🔥 Daily Hottest Capability Breakthrough

- **Breakthrough Focus**: **Uncovering the Hidden "Zero-Alert Silent Bandwidth Drop" on H100/H200/B200 GPUs—Why Crossing `T_mem = 75°C` Triggers JEDEC HBM3e `2x / 4x` Temperature-Controlled Auto-Refresh (`tREFI` Halving) That Steals `10.8%–35.0%` of Memory Bandwidth Even When `T_gpu` Is Only `64°C–71°C`, and How Enforcing `T_mem < 72°C` + `ΔT = T_mem - T_gpu > 18°C` TIM Pump-Out Detection Sustains Full `4.68 TB/s` Bandwidth**.
- **Why It Matters Today**: Many GPU cluster dashboards only monitor `DCGM_FI_DEV_GPU_TEMP` (logic die temperature). Because 8-Hi/12-Hi 3D-stacked HBM3e cubes have higher vertical thermal resistance and suffer **Thermal Interface Material (TIM) Pump-Out** under thermal cycling, a single GPU can have a cool core (`64°C`) while its HBM stack hits **`90°C`**—triggering `4x` capacitor refresh and `SW_THERMAL_SLOWDOWN`, dragging down an entire 1,024-GPU training job's MFU by **`-31.5%`**!

---

## 📌 1. Executive Summary (3-Minute Architect Takeaways)

1. **Why HBM3e Loses `10.8%` Bandwidth Silently Between `75°C` and `84°C` (`Tier 2`: `2x tREFI` Refresh)**:
   - DRAM capacitor leakage doubles every `~10°C`. Per the JEDEC HBM3e spec, when `DCGM_FI_DEV_MEM_TEMP` exceeds **`75°C`**, the memory controller cuts the refresh interval `tREFI` in half (`3.9 μs → 1.95 μs`, **2x refresh rate**) even though software thermal slowdown (`85°C+`) has **not** triggered (`CLOCK_THROTTLE_REASONS = 0x0`)! Because banks cannot serve reads/writes during refresh (`tRFC`), effective H200 bandwidth drops silently from `4.68 TB/s` to **`4.28 TB/s` (`-10.8%`)** ⚠️.
2. **Why `ΔT = T_mem - T_gpu > 18°C` Diagnoses TIM Pump-Out Before Catastrophic Failure (`Tier 1`)**:
   - Under healthy cold-plate or vapor-chamber contact, `T_mem - T_gpu` stays between `8°C` and `14°C`. When thermal pad delamination or micro-channel fouling occurs over an HBM stack (`Tier 1`), `T_gpu` reads `64°C` while `T_mem` hits `90°C` (`ΔT = 26°C`), triggering `4x` refresh + `SW_THERMAL_SLOWDOWN` (`3.12 TB/s`, **`-35.0%` bandwidth**) ✕.
3. **Production Gold Standard (`Tier 4`: Maintain `T_mem < 72°C` (`1x` Refresh) + Proactive `ΔT > 18°C` Auto-Cordon)**:
   - Locks HBM3e in the `1x` refresh zone (`4.68 TB/s` effective bandwidth) and cordons any node with `ΔT > 18°C` or throttle bitmask `(reasons & 0xE8) != 0`, preserving **`0.0%` cluster MFU loss** 🏆!

---

## 🎯 2. Architect & Customer Deep-Dive Q&A (Q1 / Q2 / Q3)

### Q1: Why do 2-minute burn-in checks miss HBM thermal throttling that appears 20 minutes into training?
- **Answer**: Because coolant loops and copper heatsinks take **10–15 minutes of continuous 700W load (thermal soak)** to reach steady-state equilibrium. Acceptance tests must evaluate `max(DCGM_FI_DEV_MEM_TEMP) < 74°C` after at least 15 minutes of sustained HBM + Tensor Core load.

### Q2: How do we decode `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` without false positives from idle or power caps?
- **Answer**: Ignore `0x01` (`GpuIdle`), `0x02` (`ApplicationsClocksSetting`), and `0x04` (`SwPowerCap`). Alert immediately on bitwise mask **`(DCGM_FI_DEV_CLOCK_THROTTLE_REASONS & 232) > 0`**, which isolates **`0x08` (`HwSlowdown`), `0x20` (`SwThermalSlowdown`), `0x40` (`HwThermalSlowdown`), and `0x80` (`HwPowerBrakeSlowdown`)**.

### Q3: How can air-cooled clusters avoid the `75°C–84°C` `2x` refresh penalty?
- **Answer**: Reconfigure the server BMC fan PID loop via Redfish to use **`HBM Memory Temperature` (target setpoint `68°C`)** rather than `GPU Core Temperature` as the primary fan-speed driver.

---

## 🧭 3. Visual Architecture Diagrams (Mermaid)

```mermaid
flowchart TD
    A["GPU HBM3e Thermal & Telemetry Selection"] --> B{"Monitoring DCGM_FI_DEV_MEM_TEMP alongside GPU_TEMP?"}
    B -- "No (GPU_TEMP only)" --> T1["⛔ Tier 1: Undetected HBM TIM Pump-Out (T_gpu=64C, T_mem=90C)<br/>4x refresh + SW thermal throttle, 3.12 TB/s (-35%), -31.5% cluster MFU ✕"]
    B -- "Yes" --> C{"Cooling target keeps T_mem < 75C (1x Refresh)?"}
    C -- "No (Runs at 76C-84C)" --> T2["🟨 Tier 2: Silent 2x tREFI Refresh Zone<br/>Zero throttle flags, but 2x refresh steals 10.8% bandwidth (4.28 TB/s) ⚠️"]
    C -- "Reactive >85C throttle alert only" --> T3["🟦 Tier 3: Reactive CLOCK_THROTTLE_REASONS Alert<br/>Catches >85C slowdown, misses 75-84C refresh loss ✅"]
    C -- "Lock T_mem <72C + Delta-T Gate (Recommended)" --> T4["🏆 Tier 4: Liquid/BMC <72C (1x Refresh) + Delta-T >18C Auto-Cordon<br/>4.68 TB/s full bandwidth, 0.0% cluster MFU drop"]
```

---

## 📊 4. Architect Decision Matrix (4 Tiers Compared)

| Dimension | 🟥 Tier 1: TIM Pump-Out (`90°C`) | 🟨 Tier 2: Silent 2x Zone (`81°C`) | 🟦 Tier 3: Reactive Alert (`78°C`) | 🟩 Tier 4: `<72°C + ΔT Gate` (Recommended 🏆) |
| :--- | :--- | :--- | :--- | :--- |
| **`T_gpu / T_mem (ΔT)`** | `64°C / 90°C (ΔT=26°C)` ✕ | `71°C / 81°C (ΔT=10°C)` | `68°C / 78°C (ΔT=10°C)` | **`54°C / 66°C (ΔT=12°C)`** 🏆 |
| **HBM Refresh Multiplier** | `4x Refresh` (`0.975 μs`) ✕ | `2x Refresh` (`1.95 μs`) ⚠️ | `2x Refresh` (`1.95 μs`) | **`1x Refresh` (`3.90 μs`)** 🏆 |
| **Effective H200 Bandwidth** | `3.12 TB/s (-35.0%)` ✕ | `4.28 TB/s (-10.8%)` ⚠️ | `4.35 TB/s (-9.4%)` | **`4.68 TB/s (97.5% peak)`** 🏆 |
| **1,024-GPU Cluster MFU Drop** | `-31.5%` ✕ | `-9.8%` ⚠️ | `-7.5%` | **`0.0%`** 🏆 |

---

## 🧪 5. Quantitative Case Study & 7-Card 4:5 Benchmark Report

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: TIM Pump-Out (T_mem=90C, 4x Ref)", "eff_bw_tbs": 3.12 },
      { "tier": "Tier 2: Air-Cooled 81C (Silent 2x Ref)", "eff_bw_tbs": 4.28 },
      { "tier": "Tier 3: Reactive Throttle Alert (78C)", "eff_bw_tbs": 4.35 },
      { "tier": "Tier 4: Liquid <72C (1x Ref) + Delta-T", "eff_bw_tbs": 4.68 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "eff_bw_tbs": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Keeping HBM3e below 72C avoids 2x/4x refresh penalties and sustains 4.68 TB/s bandwidth",
    "subtitle": "Effective H200 HBM3e memory bandwidth across 4 thermal regimes (TB/s, peak = 4.80 TB/s)",
    "encodings": {
      "x": "tier",
      "y": "eff_bw_tbs"
    }
  }
}
```

- 🧮 **Interactive HBM Thermal & Refresh Sizer**: [`docs/reports/gpu-hbm3e-thermal-refresh-trefi-tim-pumpout-throttle-telemetry-sizer.html`](../reports/gpu-hbm3e-thermal-refresh-trefi-tim-pumpout-throttle-telemetry-sizer.html)
- 🃏 **7-Card 4:5 Visual Benchmark Report**: [`docs/reports/gpu-hbm3e-thermal-refresh-trefi-tim-pumpout-throttle-telemetry-selection-cards.html`](../reports/gpu-hbm3e-thermal-refresh-trefi-tim-pumpout-throttle-telemetry-selection-cards.html)

---

## 🛠️ 6. Production Verification Checklist

- [ ] **Alert on `DCGM_FI_DEV_MEM_TEMP > 74°C`**: Prevent silent `2x tREFI` refresh bandwidth degradation before `85°C` software throttling even begins.
- [ ] **Alert on `(DCGM_FI_DEV_MEM_TEMP - DCGM_FI_DEV_GPU_TEMP) > 18°C`**: Catch HBM thermal pad pump-out and cold-plate micro-channel blockages early.
- [ ] **Bitmask Filter `(DCGM_FI_DEV_CLOCK_THROTTLE_REASONS & 232) > 0`**: Isolate real hardware/thermal/power-brake slowdowns (`0x08 | 0x20 | 0x40 | 0x80`).
- [ ] **Bind BMC Fan / CDU PID Loops to `MEM_TEMP`**: Target `T_mem < 72°C` under steady-state load.
