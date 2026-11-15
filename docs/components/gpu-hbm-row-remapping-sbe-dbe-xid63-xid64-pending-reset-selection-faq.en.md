# GPU HBM3/HBM3e Hardware `Row Remapping` (`XID 63 / XID 64`, `REMAPPED_ROWS_PENDING` & `ROW_REMAP_FAILURE`): Why Un-Reset Pending Remaps Trigger Repeat `DBE` Crashes & Automated Drain-Reset-RMA Selection Guide

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Grounded in [`experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/results.json`](../../experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/results.json).
> **中文版本**: [GPU HBM3/HBM3e 硬件级显存自愈（`Row Remapping` 行重映射）与 `XID 63 / XID 64` 运维决策：为什么出现 `REMAPPED_ROWS_PENDING > 0` 后不执行 `nvidia-smi -r` 重置 GPU 会导致同一行反复引爆二次 `DBE` 崩溃？`SBE` 累积风暴与 `ROW_REMAP_FAILURE` 硬件 RMA 报修红线选型指南](./gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-selection-faq.md)

![GPU HBM3/HBM3e Row Remapping XID 63 vs XID 64 Summary](../../assets/svg/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **Eliminating `88.0%` Repeat Bad-Row Job Crashes and `92.0%` Unnecessary Hardware RMAs on `H100 / H200 / B200` Clusters by Mastering the Two-Phase Silicon Lifecycle of HBM3/HBM3e `Row Remapping` (`XID 63 REMAPPED_ROWS_PENDING = 1 -> 8.0s In-Band nvidia-smi -r Activation` vs. `XID 64 ROW_REMAP_FAILURE = 1` Permanent RMA Cordon, `Tier 4`)**: On Ampere, Hopper, and Blackwell GPUs, hardware **HBM Row Remapping** replaces degraded HBM rows with factory-provisioned spare rows inside each HBM bank. However, misunderstanding its two-phase activation causes two major operational failures:
  1. **Tier 1 — Rescheduling Jobs While `REMAPPED_ROWS_PENDING == 1` (`XID 63`) Without Running `nvidia-smi -r` (`88.0%` Repeat Crash Rate!)**:
     - When an HBM row suffers a Double-Bit ECC error (`DBE`) or crosses the Single-Bit ECC (`SBE`) threshold, the GPU logs **`XID 63`**, records the faulty row address in non-volatile `InfoROM`, and sets **`REMAPPED_ROWS_PENDING = 1` (`DCGM_FI_DEV_ROW_REMAP_PENDING`)**.
     - **Critical Hardware Law**: The faulty HBM row is **NOT** physically swapped out while the memory controller is running! The hardware MUX only activates the spare HBM row **during the next `GPU Reset` (`nvidia-smi -r` or reboot)**!
     - Rescheduling a new training job onto the GPU while `REMAPPED_ROWS_PENDING == 1` allocates tensors right back onto the un-remapped broken row (`88.0%` repeat crash rate)!
  2. **Tier 2 — Blindly RMA-ing Every GPU That Logs `XID 63` or `REMAPPED_ROWS_DBE > 0` (`92.0%` False RMA Rate, `7 Days` Lost Capacity)**:
     - HBM3/HBM3e stacks include dedicated spare rows per bank specifically to heal isolated cell failures permanently at zero performance or capacity loss. Sending healthy self-healed GPUs for RMA wastes weeks of cluster availability!
  3. **Tier 3 — Resetting on `Pending`, But Ignoring `XID 64` (`ROW_REMAP_FAILURE == 1`) When a Bank's Spare Rows Are Exhausted (`14.5%` Repeat Crash Rate)**:
     - Once a specific HBM bank exhausts all of its spare rows, the GPU logs **`XID 64`** and sets **`ROW_REMAP_FAILURE = 1`**. At this point, no reset can heal the bank!
  4. **Tier 4 — Closed-Loop `Row Remapping` State Machine (`Pre-Flight Gate on PENDING == 0` + `8.0s Automated In-Band nvidia-smi -r on XID 63` + `Immediate RMA Cordon on XID 64 / ROW_REMAP_FAILURE == 1` — `0.0%` Repeat Crashes, `0.0%` False RMAs, `8.0s` Recovery)**:
     - Blocks scheduling whenever `PENDING == 1`, executes an `8.0s` in-band `nvidia-smi -r` to activate the spare HBM row, verifies `PENDING == 0 && FAILURE == 0`, and cordons for hardware RMA **only** when `ROW_REMAP_FAILURE == 1` (`XID 64`) or when `DBE` remaps spike (`>= 4` in 24h)!

---

## 1. 📌 Executive Summary

1. **`XID 63` (`REMAPPED_ROWS_PENDING = 1`) Means "Reset Me (`nvidia-smi -r`) to Heal in 8 Seconds"**: Never schedule a new job while `remapped_rows.pending == 1`, and never RMA a card solely because `remapped_rows.unc > 0` if `remapped_rows.failure == 0`!
2. **`XID 64` (`ROW_REMAP_FAILURE = 1`) Means "Spare Rows Exhausted — Cordon for Hardware RMA Immediately"**: Once `remapped_rows.failure == 1`, the HBM bank has zero remaining spare rows and must be replaced.

---

## 2. 🎯 Core Architect Q&A (Q1 / Q2 / Q3)

### Q1: How does HBM `Row Remapping` (`A100 / H100 / B200`) differ from legacy `Dynamic Page Blacklisting` (`V100`)?
- **Answer**: Legacy Page Blacklisting masked out `4KB / 64KB` regions in software inside the OS driver, shrinking usable VRAM capacity and fragmenting contiguous allocations. Hardware **Row Remapping** redirects the row address inside the HBM memory controller silicon to a factory spare row upon GPU reset—preserving **`100.0%` contiguous HBM capacity and bandwidth**!

### Q2: What exact command should run in every Slurm `Prolog` / Kubernetes `InitContainer` pre-flight check?
- **Answer**:
  ```bash
  nvidia-smi --query-gpu=index,remapped_rows.pending,remapped_rows.failure --format=csv,noheader
  ```
  If `remapped_rows.failure != 0`, drain the node for RMA. If `remapped_rows.pending != 0`, trigger `nvidia-smi -r` before admitting the workload!

---

## 3. 🧭 Visual Diagrams

```mermaid
flowchart LR
    A["Tier 1: Un-Reset PENDING=1 (88.0% Repeat Crash ✕)"] --> B["Tier 2: Blind RMA on XID 63 (92.0% False RMA · 7d Downtime ✕)"]
    B --> C["Tier 4: Auto 8s nvidia-smi -r on XID 63 + RMA Gate on XID 64 (0% Crash · 0% False RMA · 8.0s 🏆)"]
```

---

## 4. 📊 4-Tier Architect Selection Matrix

| Metric | Tier 1: Un-Reset `PENDING=1` | Tier 2: Blind RMA on `XID 63` | Tier 3: Reset Without `XID 64` Gate | Tier 4: **Auto `8s` Reset (`XID 63`) + RMA Gate (`XID 64`)** |
| :--- | :--- | :--- | :--- | :--- |
| **Repeat Bad-Row Crash Rate** | **`88.0%` (Row Still Active!)** ✕ | `0.0%` | `14.5%` (Exhausted Bank) ⚠️ | **`0.0%` (Spare Row Activated in Silicon)** 🏆 |
| **Unnecessary False RMA Rate** | `0.0%` | **`92.0%` (Healthy Cards RMA'd)** ✕ | `5.0%` | **`0.0%` (Uses Built-In HBM Spare Rows)** 🏆 |
| **Healing Downtime (`Seconds`)** | `7,200.0 s` ✕ | **`604,800.0 s` (`7 Days` RMA)** ✕ | `1,800.0 s` | **`8.0 s` (Automated In-Band `nvidia-smi -r`)** 🏆 |

---

## 5. 🧪 Quantitative Reference & Tooling

- **Evaluator & Sealed Data**: [`experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/run.py`](../../experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/run.py) · [`results.json`](../../experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/results.json)
- **Interactive Sizer**: [`docs/reports/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-sizer.html`](../reports/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-sizer.html)
- **7-Card 4:5 Visual Report**: [`docs/reports/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-selection-cards.html`](../reports/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & Production Checklist

- **✅ Good fit**: All `A100 / H100 / H200 / B200` clusters enforcing pre-flight `remapped_rows.pending == 0` gates, automated `nvidia-smi -r` activation on `XID 63`, and RMA cordoning on `XID 64` (`remapped_rows.failure == 1`).
- **⛔ Not for**: Admitting workloads onto GPUs with `remapped_rows.pending == 1` or RMA-ing cards after isolated successful row remaps.
- [ ] Gate job admission on `DCGM_FI_DEV_ROW_REMAP_PENDING == 0` and `DCGM_FI_DEV_ROW_REMAP_FAILURE == 0`.
- [ ] Automate in-band `nvidia-smi -r` when `PENDING == 1` and permanent node drain when `FAILURE == 1` (`XID 64`).
