# Round128: optional NMS scalar presence and class boundaries

Original exact rational one-batch greedy NMS with independent
candidate-removal selection and integer-cell area oracles.
Synthetic,independently_reproduced=false;no native ONNX/GPU/quality.

```powershell
python -m experiments.nms_optional_contract_selection.run --check
python -m unittest discover -s tests -p test_nms_optional_contract_selection.py -q
```

`--measure` deliberately replaces original CPU float samples:
cachedcanonicalgeometry versus pairwise normalization,
disjoint/allselected;setup/sorting/score/native/GPUexcluded.

Selected strict explicit score predicate is pinned-reference
behavior,not universal backend equality/tie certification.
Absent score differs fromzero;absentcapdefaultszero.
Originalsupports scalar/one-itemlist/empty/None and rejects malformed
wrappers. Corner/centergeometry,ties andperclassindicesexplicit.

ONNXpin`3c41120b9c69657a1dd13407b080bd451ed06c9c`,
Oct28fix`70f6a89920a3a17d0ea1cd84c0812a90d1e566e2`,
Apache-2.0. Original implementation/oracles/art,no code copied.

Figure-tierclass-ticket cabinet:original60solids,
base/fourfeet/threetrays/sixrails/21cards/21tabs/fourfasteners.
Staticclass/presence/indexmetaphor;browserinspectionnotperformed.

- [中文](../../docs/components/nms-optional-scalars-per-class-iou-boundary-selection-faq.md)
- [English](../../docs/components/nms-optional-scalars-per-class-iou-boundary-selection-faq.en.md)
- [H5](../../docs/reports/nms-optional-scalars-per-class-iou-boundary-sizer.html)
- [7 cards](../../docs/reports/nms-optional-scalars-per-class-iou-boundary-selection-cards.html)
- [Report](report.json),[CPU records](latency.json)
