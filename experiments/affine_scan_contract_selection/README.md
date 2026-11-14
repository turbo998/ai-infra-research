# Round127: ordered affine scan,state reset and chunk carry

Original exact scalar recurrence,buffered Hillis-Steele prefix and
independent product-expansion oracle. Synthetic,
independently_reproduced=false; separately measured original CPU
float paths are not native Mamba/GPU benchmarks.

```powershell
python -m experiments.affine_scan_contract_selection.run --check
python -m unittest discover -s tests -p test_affine_scan_contract_selection.py -q
```

`--measure` deliberately replaces CPU samples.
Serial Python recurrence versus extra-round/copy Python tree;
not parallel wall-time prediction. Native selective coefficients,
discretization,gates,projection,complex lanes/backward excluded.
Exact associativity does not establish floating bitwise parity.

Explicit zero-reset packing is original extension,not native reset
API assertion. Identity(1,0),not(0,0). Carry actual state across chunks.

Mamba pin`e9594ce1c732d97440f0332fdc43170a2294dbfa`,Apache-2.0.
Classic header lastJune2024ROCm;latestJuly2026Mamba3Blackwell
fix is different kernel,not executed. Original code/oracle/art,
no implementation copied.

Figure-tier ordered-state ticket sorter: original60solids,
base/fourfeet/threetrays/sixrails/21cards/21tabs/fourfasteners.
Static ordered-prefix/segment metaphor;browser inspection not performed.

- [中文](../../docs/components/ordered-affine-prefix-scan-segment-reset-selection-faq.md)
- [English](../../docs/components/ordered-affine-prefix-scan-segment-reset-selection-faq.en.md)
- [H5](../../docs/reports/ordered-affine-prefix-scan-segment-reset-sizer.html)
- [7 cards](../../docs/reports/ordered-affine-prefix-scan-segment-reset-selection-cards.html)
- [Report](report.json),[CPU records](latency.json)
