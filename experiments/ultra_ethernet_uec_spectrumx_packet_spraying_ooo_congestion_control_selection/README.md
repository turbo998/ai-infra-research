# Next-Gen AI Ethernet (NVIDIA Spectrum-X & Ultra Ethernet UEC): Packet Spraying, OOO Direct Data Placement & CC Benchmark

Deterministic quantitative evaluation model for [`docs/components/ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control-selection-faq.md`](../../docs/components/ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control-selection-faq.md) and [`.en.md`](../../docs/components/ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/ultra_ethernet_uec_spectrumx_packet_spraying_ooo_congestion_control_selection/run.py
python experiments/ultra_ethernet_uec_spectrumx_packet_spraying_ooo_congestion_control_selection/run.py --check
python -m unittest tests/test_ultra_ethernet_uec_spectrumx_packet_spraying_ooo_congestion_control_selection.py
```
