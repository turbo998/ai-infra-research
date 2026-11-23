# Milestone `#525`: Enterprise AI Datacenter Direct-to-Chip Liquid Cooling (`DLC >45kW–130kW/Rack`), `PUE` & Thermal Throttling Sizer

Reproducible analytical model comparing **Legacy Air Cooling (`PUE 1.54`, `18.5%` GPU thermal throttling loss, `32` spread-out racks)**, **Rear-Door Heat Exchanger (`RDHx`, `PUE 1.32`)**, and **Direct-to-Chip Cold-Plate Liquid Cooling (`DLC + Redundant CDU`, `120 kW/rack`, `PUE 1.12`, `0.0%` thermal throttling, `8` compact racks enabling short DAC/AOC links, `-45.4%` 3-year power/cooling TCO 🏆)**.

## Run

```bash
python experiments/enterprise_milestone_525_dlc_liquid_cooling_pue_thermal_throttling_bible/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-milestone-525-dlc-liquid-cooling-pue-thermal-throttling-bible-faq.md)
  - [English FAQ](../../docs/components/enterprise-milestone-525-dlc-liquid-cooling-pue-thermal-throttling-bible-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/milestone-525-dlc-liquid-cooling-pue-thermal-throttling-lab.html](../../docs/h5/milestone-525-dlc-liquid-cooling-pue-thermal-throttling-lab.html)
