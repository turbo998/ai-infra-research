# Enterprise High-Density AI Data Center Power Delivery, Synchronous `di/dt` Transients, Capacitive/BESS Smoothing & Liquid Cooling (`RDHx` vs. `L2A` vs. `L2L`) Retrofit Sizer (`#668`)

Reproducible analytical model quantifying why synchronous GPU barrier/AllReduce transitions create a `75.0%` step-load power swing (`264 kW → 1,056 kW` in `<20 ms`, plus `118%` peak excursion) that trips standard diesel generators and UPS systems, why naive static `70%` GPU power capping sacrifices `19.4%` throughput, how **3-Layer Power Smoothing (`ORv3 48V Supercapacitor Shelf + Rack Li-Ion BESS + Synthetic Idle Floor`)** compresses utility step swing to `11.5%` with only `0.8%` throughput loss, and how to select between **`RDHx` (`45 kW/rack`), `L2A` Sidecar CDU (`75 kW/rack` with zero facility water plumbing), and `L2L` Direct-to-Chip DLC (`140 kW/rack`, `PUE 1.11`)**.

```bash
python experiments/enterprise_ai_datacenter_power_transient_bess_dlc_retrofit_sizer/run.py
pytest tests/test_enterprise_ai_datacenter_power_transient_bess_dlc_retrofit_sizer.py -q
```
