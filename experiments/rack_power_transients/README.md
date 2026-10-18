# 130kW–600kW Liquid-Cooled AI Rack Power Transients (`dP/dt`), OCP ORv3 Supercap CBU Sizing & DCGM Power Smoothing Co-Design

## Overview

This deterministic Python standard-library experiment models the electrical power step transients (`dP/dt`) generated when a `16,384`-GPU cluster transitions synchronously between Tensor Core GEMM compute (`1200–3200W/GPU`) and collective communication barriers (`280–650W/GPU`) within `3.0–5.0 ms` across four rack generations (`GB200 NVL72 132kW`, `GB300 NVL72 180kW`, `Vera Rubin NVL144 300kW`, and `Rubin Ultra NVL144 600kW`), and compares four mitigation strategies:

1. **Unmitigated Synchronized Step (`unmitigated`)**: Direct `3–5 ms` power step imposed on rack PSUs, UPS inverters, and utility substation transformers.
2. **NVML/DCGM Firmware Power Smoothing Only (`DCGM_FI_DEV_PWR_SMOOTHING_*`)**: Enforces ramp-down hysteresis (`DCGM_FI_DEV_PWR_SMOOTHING_PROFILE_RAMP_DOWN_HYST_VAL = 15 ms`) and controlled ramp rates (`DCGM_FI_DEV_PWR_SMOOTHING_PROFILE_RAMP_UP_RATE`), trading step-time slowdown (`3.18–3.50%`) for reduced swing.
3. **Collective-Wait Dummy Burn Kernel Only (`collective_dummy_burn`)**: Launches synthetic register-resident FP16/BF16 FMA loops on idle SMs during NCCL collective waits to maintain a `75%` TDP floor (`DCGM_FI_DEV_PWR_SMOOTHING_APPLIED_TMP_FLOOR`), incurring `5.02–7.40%` extra facility energy burn.
4. **Hybrid Co-Design (`orv3_cbu_plus_fw_smoothing`)**: Combines OCP ORv3 48V Supercapacitor Capacitor Backup Units (`CBU`, `48V -> 36V` `43.75%` usable energy fraction) with firmware ramp-rate control to clamp rack ramp rates to `50 kW/s` (`11.38 MW/s` across `16,384` GB200 GPUs) at only `0.32%` ESR loss.

## Reproduction

```bash
python experiments/rack_power_transients/run.py
python experiments/rack_power_transients/run.py --check
```

## Key Findings (from `results.json`)

- **Unmitigated Cluster Slew-Rate Shock**: Across `16,384` GPUs (`227.6` GB200 NVL72 racks), a `5.0 ms` transition across a `66.2 kW/rack` swing (`132.0 kW` peak to `65.8 kW` trough) generates **`3,014.7 MW/s`** instantaneous `dP/dt` (`15.07 MW` step in `5 ms`), rising to **`13,926.4 MW/s`** (`367.2 kW/rack` swing in `3.0 ms`) on `600 kW` Rubin Ultra racks.
- **OCP ORv3 48V Supercapacitor CBU Sizing**: Clamping rack `dP/dt` to `<= 50 kW/s` (`264.91x` cluster slew-rate reduction to `11.38 MW/s`) requires **`43.88 kJ`** of usable energy per `132 kW` GB200 rack (**`87.1 Farads`** at `48V` nominal / `36V` minimum cutoff) and **`274.23 kJ`** (**`544.1 Farads`**) per `300 kW` Rubin NVL144 rack.
- **Dummy Burn vs Firmware Smoothing vs Hybrid CBU Trade-off**: Maintaining a `75%` TDP floor via Dummy Burn wastes **`5.88%`** (`GB200`) to **`7.40%`** (`Rubin Ultra`) additional cluster energy, while pure firmware ramp-rate limiting without CBU buffering costs **`3.18–3.50%`** training step time due to `SW_POWER_CAP` clock throttling during GEMM ramp-up.
