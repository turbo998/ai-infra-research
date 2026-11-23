# Enterprise Construction & Infrastructure BIM/CAD Quantity Takeoff: `Manual QS` vs. `Full-Sheet Raster 72B VLM` vs. `Vector DXF/PDF Parser + Tile VLM + Deterministic Shapely Geometry Engine` (`#558`)

- **Chinese FAQ**: [`../../docs/components/enterprise-construction-bim-cad-quantity-takeoff-geometry-sizing-faq.md`](../../docs/components/enterprise-construction-bim-cad-quantity-takeoff-geometry-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-construction-bim-cad-quantity-takeoff-geometry-sizing-faq.en.md`](../../docs/components/enterprise-construction-bim-cad-quantity-takeoff-geometry-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/construction-bim-cad-quantity-takeoff-geometry-lab.html`](../../docs/h5/construction-bim-cad-quantity-takeoff-geometry-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-construction-bim-cad-quantity-takeoff-geometry-sizing-sizer.html`](../../docs/reports/enterprise-construction-bim-cad-quantity-takeoff-geometry-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-construction-bim-cad-quantity-takeoff-geometry-sizing-cards.html`](../../docs/reports/enterprise-construction-bim-cad-quantity-takeoff-geometry-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_construction_bim_cad_quantity_takeoff_geometry_sizing/run.py
python -m pytest tests/test_enterprise_construction_bim_cad_quantity_takeoff_geometry_sizing.py -q
```
