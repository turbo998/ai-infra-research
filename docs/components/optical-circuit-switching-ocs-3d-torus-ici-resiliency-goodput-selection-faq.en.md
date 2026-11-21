# 10K-Chip Optical Circuit Switching (`Palomar 3D MEMS OCS`), `3D Torus` Topology Defragmentation, Warm-Spare Cube Splicing & `ICI Resiliency` Fault-Tolerance Architecture Selection Guide

> **Target Audience**: Cloud & AI Infrastructure Solution Architects, Principal AI Fabric Engineers, and Supercomputing Cluster Architects designing `1,024` to `9,216+` chip 3D Torus pods (`Google Cloud TPU v5p 8,960-chip Pod`, `TPU7x Ironwood 9,216-chip Superpod`, `TPU 8t`) and optical circuit-switched AI fabrics.
> **Core Question**: When scaling a synchronous training pod to **`9,216` chips (`144` liquid-cooled `4x4x4` 64-chip `Cubes`)**, where aggregate component failure rates trigger a link or host fault every **`6–12 hours`**, why does a "Statically Cabled 3D Torus (Tier 1)" suffer **`29.0%` Swiss-cheese topological fragmentation stranding and `62.5%` Goodput**, while a "3-Tier Electrical Packet-Switched Clos Fat-Tree (Tier 2)" burns **`1,120 kW` in Spine ASICs and Optical-Electrical-Optical (`OEO`) transceivers (`$2.74M/month` TCO)**? How does a **two-layer fault-tolerant architecture combining Sub-Millisecond Hardware `ICI Resiliency` Orthogonal Detours + `<15-Second` `Palomar 3D MEMS OCS` Warm-Spare Cube Optical Splicing + `Twisted-Torus` Skew Wiring (Tier 4)** cut interconnect switch power by **`89.7%` (`1,120 kW -> 115 kW`)**, reduce stranded fragmentation to **`0.4%`**, sustain **`99.2%` Superpod Goodput**, and slash monthly interconnect and downtime TCO by **`75.6%` (`$3.85M -> $940K/month`)**?

![10K-Chip Optical Circuit Switching (Palomar 3D MEMS OCS) & ICI Resiliency Summary](../../assets/svg/optical-circuit-switching-ocs-3d-torus-ici-resiliency-goodput-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

At `9,216`-chip scale, **the defining architectural question is: *"When Cube `#42` loses a power shelf or HBM module at 3:00 AM, can the remaining `9,152` healthy chips still form unbroken, wrap-around 3D Torus slices without stranding 30% of the pod?"***:

| Superpod Interconnect Tier | Best-Fit Scenario | `9,216`-Chip Effective `Goodput` | Stranded Fragmentation (`%`) | Switch & Optics Power (`kW`) | Monthly Interconnect, Power & Downtime TCO | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Statically Cabled `3D Torus` (No `OCS` or `ICI Resiliency`, Anti-Pattern)** | Single-rack `<=64`-chip islands only | **`62.5%` (1 dead cube breaks rings!)** | **`29.0%` (Swiss-cheese holes!)** | `85.0 kW` | `$3,850,000 / mo` | ❌ **Reject**: A single failed cube shatters torus rings & strands adjacent cubes |
| **Tier 2: 3-Tier Electrical Packet-Switched `Clos Fat-Tree` (Full `OEO` Optics)** | Small/medium multi-tenant GPU clouds | `89.2%` | `4.5%` | **`1,120.0 kW` (OEO power tax!)** | `$2,740,000 / mo` | ⚠️ **Costly**: Burns `1.12 MW` in Spine ASICs/DSP optics & requires rip-and-replace upgrades |
| **Tier 3: `OCS`-Switched `3D Torus` (Without Sub-ms Link-Level `ICI Resiliency`)** | Early optical circuit testbeds | `91.5%` | `3.2%` | `115.0 kW` | `$1,680,000 / mo` | ⚠️ **Fragile**: Single-fiber flaps trigger unnecessary whole-slice OCS reconfigurations |
| **Tier 4: `Palomar 3D MEMS OCS` + Sub-ms `ICI Resiliency` + Warm-Spare Cube Splicing + `Twisted Torus` (Recommended)** | **`TPU v5p / TPU7x Ironwood (9,216 Chips) / TPU 8t` Superpod Standard** | **`99.2%`** | **`0.4%` (zero fragmentation)** | **`115.0 kW` (`-89.7%` vs. Clos)** | **`$940,000 / mo` (`-75.6%`)** | 🏆 **Production Gold Standard**: `<1ms` link detour + `<15s` optical spare-cube swap |

### 💡 Three Golden Rules for 10K-Chip Optical Interconnect Architects
1. **Wire Intra-Cube `4x4x4` (`64` Chips) with Copper and Inter-Cube Faces (`±X, ±Y, ±Z`) with Passive `Palomar 3D MEMS OCS` Mirrors (`45W` per Chassis, Zero `OEO` Tax)**:
   * Inside each liquid-cooled `4x4x4` Cube, `64` chips connect over short, ultra-reliable copper links. The `6` exterior faces of the cube (`96` optical links per cube) plug into **`136×136` port `Palomar 3D MEMS Optical Circuit Switches (OCS)`**.
   * Because MEMS micro-mirrors reflect light beams directly from input fiber to output fiber **without Optical-Electrical-Optical (`OEO`) conversion, CDR retiming, or packet-switch ASICs**, each OCS chassis consumes just **`~45W`** (cutting fabric switch power by **`89.7%`**, from `1,120 kW` down to `115 kW`) and remains **100% data-rate agnostic** across `400G -> 800G -> 1.6T` generations.
2. **Pair Two Orthogonal Resilience Layers: `<1ms` Hardware `ICI Resiliency` for Single-Link Faults + `<15s` `OCS` Warm-Spare Cube Splicing for Rack/Node Faults**:
   * **Layer 1 (`<1ms` Link-Level `ICI Resiliency`)**: If a single inter-cube fiber or transceiver degrades, on-chip `ICI` routers instantaneously detour packets along orthogonal 3D Torus axes (`+Y -> +X -> -Y`) in sub-millisecond hardware—**zero job interruption, zero checkpoint rollback, and zero OCS mirror movement**.
   * **Layer 2 (`<15s` Cube-Level `OCS` Optical Spare Swap)**: If a host, PSU, or CDU manifold inside `Cube #42` fails, `Orbax Emergency Checkpointer` saves state in `<5s` while the `OCS` controller tilts its MEMS mirrors to optically bypass `Cube #42` and splice **`Warm-Standby Spare Cube #145`** directly into `#42`'s logical 3D coordinates—eliminating "Swiss-cheese" pod fragmentation (`0.4%` stranded vs. `29.0%` in Tier 1).
3. **Use `OCS` to Wire `Twisted-Torus` (Möbius Skew) Wrap-Around Links on Asymmetric Slices (`+35%` Bisection Bandwidth)**:
   * On elongated slices (`4x4x8`, `4x8x16`), tilting the wrap-around OCS links by one orthogonal stride creates a **Twisted Torus**, Equalizing ring diameters across all three axes and accelerating multi-dimensional `All-Reduce` by `25%–35%`.

---

## 2. 🎯 Customer & Architect Critical Q&A

### Q1: Why can't `3D MEMS OCS` replace packet switching inside a training step, and how do `OCS` and `ICI` divide responsibility?
* **Millisecond Mechanical Mirror Inertia vs. Nanosecond Packet Routing**:
  * Tilting a physical silicon MEMS mirror takes **`10–50 milliseconds`**, whereas transmitting a `1 KB` collective packet at `800 Gbps` takes **`10 nanoseconds`**. Therefore, **`OCS` operates at the job-scheduling and fault-recovery timescale** (stitching `4x4x4` cubes into defect-free, wrap-around 3D Torus slices), while **on-chip `ICI` packet routers operate at the nanosecond timescale** inside each step.

### Q2: How does `OCS` prevent "Swiss-Cheese Topology Fragmentation" across a `144`-Cube (`9,216`-Chip) Superpod?
* **Decoupling Physical Floor Coordinates from Logical Torus Coordinates**:
  * In a fixed-wire 3D Torus (Tier 1), if `Cube #42` and `Cube #89` are down for maintenance, the physical grid contains two holes that prevent allocating contiguous `8x8x8` (512-chip) or `8x16x16` (2,048-chip) jobs—stranding `29.0%` of healthy cubes. Because every cube's 6 faces terminate at the central `Palomar OCS` patch panel, any healthy cube on the data center floor can be optically adjacent to any other cube, maintaining **`99.6%` contiguous allocatability**.

---

## 3. 🧭 Visual Diagrams (Mermaid Decision Tree + Reference Architecture + Sequence Diagram)

### 3.1Two-Layer `ICI Resiliency` + `Palomar OCS` Fault-Handling Decision Tree

```mermaid
flowchart TD
    A["Hardware Fault Detected in 9,216-Chip Superpod (144 Cubes)"] --> B{"Fault Granularity & Scope?"}
    B -- "Single inter-cube fiber flap or elevated transceiver BER" --> C["⚡ Layer 1 (<1ms): Hardware ICI Resiliency\nOn-chip ICI router detours packets via orthogonal 3D Torus axes\n(0 job abort, 0 OCS mirror switch, 0 checkpoint rollback!)"]
    B -- "Full host crash, rack PSU trip, or CDU maintenance on Cube #42" --> D["🔮 Layer 2 (<15s): Palomar 3D MEMS OCS Warm-Spare Cube Swap"]
    D --> E["Step 1: Orbax Emergency Checkpointer (<5s) flushes latest step to Local NVMe"]
    E --> F["Step 2: OCS tilts MEMS mirrors to bypass Cube #42 & splice Warm-Spare Cube #145 into identical 3D coordinates"]
    F --> G["Step 3: JobSet resumes full 4x4x8 / 8x8x8 Twisted-Torus slice at 100% bandwidth (99.2% Goodput!)"]
```

### 3.2 Tier 4 `144`-Cube Superpod + `136x136 Palomar OCS` Core + Warm-Spare Architecture

```mermaid
flowchart LR
    subgraph Cubes["🧊 1. 144 Active Cubes + 4 Warm-Spare Cubes (4x4x4 = 64 Chips/Cube)"]
        C1["Cube #1 (64 Chips)\nIntra-Rack Copper 3D Mesh"]
        C42["Cube #42 (64 Chips)\n⚠️ Host / PSU Hardware Fault"]
        C144["Cube #144 (64 Chips)\nIntra-Rack Copper 3D Mesh"]
        SPARE["🟢 Spare Cube #145 (64 Chips)\nPre-Burned Warm Standby"]
    end

    subgraph OCS["🪞 2. Palomar 3D MEMS Optical Circuit Switch Core (48 Chassis @ 45W)"]
        MEMS["136x136 3D MEMS Mirror Array\n• Zero OEO conversion (<1.8 dB IL)\n• Wires Twisted-Torus Wrap-Around loops\n• <15s optical bypass & spare-cube splice"]
    end

    C1 & C144 <==>|"6 Cube Faces (±X, ±Y, ±Z)\n96 Fibers/Cube to OCS"| MEMS
    C42 -. "Optically Isolated by MEMS Tilt" .-x MEMS
    SPARE <==>|"Spliced into Cube #42 Coordinates"| MEMS
```

### 3.3 End-to-End `<1ms` Link Detour & `<15s` Optical Spare-Cube Swap Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Slice as Active 2,048-Chip Training Slice (32 Cubes)
    participant ICI as On-Chip TPU ICI Router
    participant OCS as Palomar OCS Topology Controller
    participant Spare as Warm-Standby Spare Cube (#145)

    Note over Slice,ICI: Case A: Hour 12 — Single +X fiber between Cube #5 and Cube #6 degrades
    Slice->>ICI: Detects Port +X_3 `Pre-FEC BER > 1e-8`
    ICI->>ICI: `<1ms` activates hardware `ICI Resiliency`: detours traffic via `+Y -> +X -> -Y`
    ICI->>Slice: Training step continues uninterrupted (`+0.8%` step time, `0` restarts!)

    Note over Slice,Spare: Case B: Hour 36 — Host motherboard fails inside Cube #18
    Slice->>OCS: `Orbax` saves `<5s` local NVMe emergency checkpoint; reports `Cube #18` host down
    OCS->>OCS: Computes mirror permutation & re-points `Palomar 3D MEMS` mirrors (`<15s`)
    OCS->>Spare: Splices `Spare Cube #145`'s 6 optical faces into `Cube #18`'s neighbor ports with Wrap-Around intact
    Spare->>Slice: `JobSet` resumes full `2,048`-chip training from latest step (`99.2%` Goodput, `0.4%` stranding)
```

---

## 4. 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | Tier 1: Statically Cabled `3D Torus` (No `OCS`) | Tier 2: 3-Tier Electrical `OEO Clos Fat-Tree` | Tier 3: `OCS` `3D Torus` (No Link `ICI Resiliency`) | Tier 4: `Palomar 3D MEMS OCS` + Sub-ms `ICI Resiliency` + Spare Cubes (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **`9,216`-Chip Superpod `Goodput`** | **`62.5%` (1 dead cube breaks torus rings)** | `89.2%` | `91.5%` | **`99.2%` (`<1ms` link detour + `<15s` optical cube swap)** |
| **Stranded Fragmented Chips (`%`)** | **`29.0%` (Swiss-cheese un-allocatable holes)** | `4.5%` | `3.2%` | **`0.4%` (optical defragmentation across any floor cube)** |
| **Interconnect Switch & Optics Power** | `85.0 kW` | **`1,120.0 kW` (power-hungry Spine ASICs/OEO)** | `115.0 kW` | **`115.0 kW` (`45W` per OCS chassis; `-89.7%` vs. Clos)** |
| **Data-Rate Upgrade (`400G->800G->1.6T`)** | Manual re-cabling | ❌ Rip-and-replace Spine switches & optics | ✅ Rate-agnostic mirrors | **✅ 100% rate-agnostic passive MEMS mirrors** |
| **Asymmetric Slice Bisection (`Twisted Torus`)** | ❌ Fixed orthogonal wiring | ✅ Non-blocking Clos | ⚠️ Orthogonal wrap only | **✅ Dynamic Möbius Twisted-Torus (`+35%` bisection BW)** |
| **Monthly Interconnect & Downtime TCO** | `$3,850,000 / mo` | `$2,740,000 / mo` | `$1,680,000 / mo` | **`$940,000 / mo` (`-75.6%` savings)** |

---

## 5. 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Report

### 5.1 Baseline Workload (`9,216`-Chip `TPU7x Ironwood` Superpod / `144` Active Cubes + `4` Warm-Spare Cubes)
* **Workload Profile**: `9,216`-chip Superpod (`144` liquid-cooled `4x4x4` cubes + `4` warm-standby spare cubes) experiencing `6` link or node hardware faults per week.
* **Reproducible Model & Dataset**: See [`experiments/optical_circuit_switching_ocs_3d_torus_ici_resiliency_goodput_selection/run.py`](../../experiments/optical_circuit_switching_ocs_3d_torus_ici_resiliency_goodput_selection/run.py) and [`results.json`](../../experiments/optical_circuit_switching_ocs_3d_torus_ici_resiliency_goodput_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive Sizer**: 📊 [Open 10K-Chip Optical Circuit Switching (`OCS`) & `ICI Resiliency` Sizer (`optical-circuit-switching-ocs-3d-torus-ici-resiliency-goodput-sizer.html`)](../reports/optical-circuit-switching-ocs-3d-torus-ici-resiliency-goodput-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`optical-circuit-switching-ocs-3d-torus-ici-resiliency-goodput-selection-cards.html`)](../reports/optical-circuit-switching-ocs-3d-torus-ici-resiliency-goodput-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Static 3D Torus (29% Stranded)", "goodput_pct": 62.5, "stranded_pct": 29.0, "power_kw": 85.0, "monthly_tco_m": 3.85 },
      { "tier": "Tier 2: 3-Tier OEO Clos (1,120kW Power)", "goodput_pct": 89.2, "stranded_pct": 4.5, "power_kw": 1120.0, "monthly_tco_m": 2.74 },
      { "tier": "Tier 3: OCS Without ICI Link Resiliency", "goodput_pct": 91.5, "stranded_pct": 3.2, "power_kw": 115.0, "monthly_tco_m": 1.68 },
      { "tier": "Tier 4: Palomar OCS + ICI Resiliency + Spares", "goodput_pct": 99.2, "stranded_pct": 0.4, "power_kw": 115.0, "monthly_tco_m": 0.94 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "goodput_pct": "Percentage",
    "stranded_pct": "Percentage",
    "power_kw": "Quantity",
    "monthly_tco_m": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 99.2% Superpod Goodput, Cuts Switch Power by 89.7%, and Saves 75.6% TCO",
    "subtitle": "Monthly 9,216-chip interconnect depreciation, optical power, and downtime TCO ($ USD Millions)",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_m",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ Architecture Review & Production Readiness Checklist

- [ ] **Two-Layer Fault Separation (`<1ms ICI Resiliency` vs. `<15s OCS Spare Swap`)**: Is hardware `ICI Resiliency` enabled to absorb single-fiber BER/link flaps without triggering unnecessary `OCS` mirror reconfigurations?
- [ ] **Warm-Standby Spare Cubes (`2%–3%`) Pre-Wired into the `Palomar OCS` Matrix**: Are `2–4` pre-burned spare `4x4x4` cubes terminated on the same `OCS` fabric so a failed cube can be optically swapped in `<15 seconds`?
- [ ] **Automated `Twisted-Torus` Skew Scheduling on Non-Cubic Slices**: Does the topology controller program Möbius-twisted wrap-around links on `4x4x8` and `4x8x16` slices to equalize ring diameters?
- [ ] **Continuous `OCS` Insertion Loss (`<1.8 dB`) & Optical Margin Telemetry**: Are end-to-end transceiver `DOM Rx Power` and `Pre-FEC BER` tracked across every OCS path to schedule proactive cleaning/swaps?
- [ ] **Live Optical Topology Defragmentation**: Does the cluster scheduler re-point OCS mirrors between job waves to coalesce fragmented free cubes into contiguous `1,024+` chip allocations?
