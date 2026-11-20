# Enterprise Manufacturing Visual Quality Inspection: <15ms Edge PatchCore/YOLO + Gray-Zone VLM Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-manufacturing-visual-quality-inspection-edge-anomaly-vlm-selection-faq.md`](../../docs/components/enterprise-manufacturing-visual-quality-inspection-edge-anomaly-vlm-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-manufacturing-visual-quality-inspection-edge-anomaly-vlm-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_manufacturing_visual_quality_inspection_edge_anomaly_vlm_selection/run.py
python experiments/enterprise_manufacturing_visual_quality_inspection_edge_anomaly_vlm_selection/run.py --check
python -m unittest tests/test_enterprise_manufacturing_visual_quality_inspection_edge_anomaly_vlm_selection.py
```
