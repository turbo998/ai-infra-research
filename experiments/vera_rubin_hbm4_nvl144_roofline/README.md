# Vera Rubin NVL144, HBM4 (2048-bit) & NVLink-6 Cross-Generation Roofline Simulator (`experiments/vera_rubin_hbm4_nvl144_roofline`)

Compares Hopper `H100 NVL8`, Blackwell `GB200 NVL72`, and Vera Rubin `VR200 NVL144` (`HBM4 2048-bit @ 13.5 TB/s`, `NVLink-6 3.6 TB/s`, `1.6T CPO`) on `1-Trillion Parameter MoE (64B active, EP=64, BS=128, L_kv=32,768 MLA)` and `1M`-context disaggregation (`2.28 ms` TPOT, `16.52x` faster than H100, and `$1.2272 / 1M tokens`, `10.04x` lower cost than H100).

## Run & Verify

```powershell
python experiments\vera_rubin_hbm4_nvl144_roofline\run.py --check
```
