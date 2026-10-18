# Rack Power, Cooling & Fault Domains (`own-offline`)

This self-contained Python standard-library model (`run.py`) evaluates four coupled infrastructure boundaries for high-density AI racks:

1. **Rack Power Density & Liquid vs. Air Cooling Thermodynamics (`evaluate_rack_cooling_thermodynamics`)**:
   - Computes total IT payload power ($P_{\text{IT}} = N_{\text{GPU}} P_{\text{GPU}} + N_{\text{CPU}} P_{\text{CPU}} + N_{\text{NVSwitch}} P_{\text{NVSwitch}} + P_{\text{aux}}$), rack wall power ($P_{\text{wall}} = P_{\text{IT}} / \eta_{\text{PSU}}$), and heat split between direct-to-chip liquid cooling ($Q_{\text{liquid}} = f_{\text{liq}} P_{\text{wall}}$) and room air ($Q_{\text{air}} = (1 - f_{\text{liq}}) P_{\text{wall}}$).
   - Evaluates required Coolant Distribution Unit (CDU) volumetric flow rate in LPM ($Q_{\text{liquid}} / (\rho_{\text{liq}} c_{p,\text{liq}} \Delta T_{\text{liq}}) \times 60$) for water vs. PG25 (25% propylene glycol) and rear-door/room airflow in CFM ($Q_{\text{air}} / (\rho_{\text{air}} c_{p,\text{air}} \Delta T_{\text{air}}) / 0.00047194745$).
   - Maps steady-state GPU junction temperature ($T_j = T_{\text{sink,out}} + P_{\text{GPU}} R_{\theta,\text{jc+coldplate}}$) against NVIDIA DCGM `dcgmlib/dcgm_fields.h` thermal thresholds (`DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS` -> `DCGM_CLOCKS_EVENT_REASON_SW_THERMAL`, `DCGM_FI_DEV_GPU_TEMP_SLOWDOWN_CELSIUS` -> `DCGM_CLOCKS_EVENT_REASON_HW_THERMAL`).
2. **Slurm `power_save.c` Token-Bucket Inrush Rate Limiting & RAPL/XCC/Cray Energy Accounting (`evaluate_slurm_power_save_and_energy`)**:
   - Reproduces `power_save_rl_setup()` (`src/slurmctld/power_save.c:L1314-1369`):
     $$\text{max\_tokens} = \max\left(1, \left\lfloor \frac{\text{rate} \times \max(1, \text{power\_save\_interval}, \text{power\_save\_min\_interval})}{60} \right\rfloor\right), \quad T_{\text{refill\_ms}} = \left\lfloor \frac{60000}{\text{rate}} \right\rfloor$$
   - Reproduces `_pick_exc_nodes()` (`src/slurmctld/power_save.c:L370-449`) partial node exclusion count (`exc_node_cnt - active_count` idle nodes kept warm).
   - Reproduces `src/plugins/acct_gather_energy/rapl/acct_gather_energy_rapl.c` 32-bit MSR counter wraparound and Xeon server DRAM $0.5^{16}\approx 15.2588\,\mu\text{J}$ unit override, plus `acct_gather_energy_xcc.c` SD650v2 trapezoidal integration error against monotonic `acct_gather_energy_pm_counters.c` Joules counters.
3. **NVIDIA DCGM `v4.6.1` Rack/MNNVL Fault Domain State Machine (`evaluate_dcgm_rack_fault_domain`)**:
   - Reproduces `modules/health/DcgmHealthWatch.cpp` health watch evaluation across `MonitorThermal`, `MonitorPower`, `MonitorCpuThermal`, `MonitorCpuPower`, `MonitorImex` (Multi-Node NVLink IMEX domain & daemon status), `MonitorNvSwitchErrorCounts`, `MonitorFabricFields`, `MonitorNVLinkErrorFields` (Blackwell+ `RECOVERY_SUCCESSFUL` = `WARN` vs `RECOVERY_FAILED` = `FAIL`), and `MonitorGpuRecoveryAction`.
4. **AI Infra Innovator Rack Power & Cooling Comparison (`evaluate_innovator_rack_power_profile`)**:
   - Compares rack-level wall power, cooling medium (water LPM vs. air CFM), and batch-1 vs. high-batch Joules/token across liquid-cooled GPU NVL72 racks, air-cooled Groq LPU racks, and liquid-cooled Cerebras CS-3 (WSE-3) cells.

## Commands

```bash
python experiments/rack_power_cooling/run.py
python experiments/rack_power_cooling/run.py --check
```
