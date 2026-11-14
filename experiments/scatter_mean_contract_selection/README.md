# Round131: ScatterND mean,negative aliases and trailing slices

Original flat integer coordinate scatter,independent destination-first
oracle and separately measured original CPU sum/count comparison.
Synthetic,independently_reproduced=false;no nativeONNX/FP/atomic/GPU.

```powershell
python -m experiments.scatter_mean_contract_selection.run --check
python -m unittest discover -s tests -p test_scatter_mean_contract_selection.py -q
```

`--measure` deliberately replaces original CPU timing records.
64destinations,grouped dense sum/count versus destination-first repeated
scan;initialization included. Not native allocator or contention speed.

ONNXpin`3c41120b9c69657a1dd13407b080bd451ed06c9c`,Apache-2.0.
VerifiedNov12meanaddition`af56232bdc6f1ecbb27a5c9d2866263d6033bc38`,
opset29:original value once,integer floor,negative aliases,
duplicate-none forbidden;int32indices supported in pinned schema.
No upstream implementation copied.

Bounded flat[M,k]tuple representation,positive shapes rank1..4,
exact Python integers;H5 rank2,none/add/mean only.
Arbitrary rankq,zero-size dimensions,FP/dtype behavior excluded.
Payload float32 width selection is not toy execution dtype.

StaticFigure-tieroriginal60-solidticketcabinet:
base,fourfeet,threetrays,sixrails,21cards,21tabs,fourfasteners.
Reusedoriginalgeometryasmean-ticketmetaphor;
no browser inspection performed after prior approval refusal.

- [中文](../../docs/components/scatternd-mean-negative-alias-slice-contract-selection-faq.md)
- [English](../../docs/components/scatternd-mean-negative-alias-slice-contract-selection-faq.en.md)
- [H5](../../docs/reports/scatternd-mean-negative-alias-slice-contract-sizer.html)
- [7 cards](../../docs/reports/scatternd-mean-negative-alias-slice-contract-selection-cards.html)
- [Report](report.json),[CPU records](latency.json)
