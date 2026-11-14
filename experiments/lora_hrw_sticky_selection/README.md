# Round132: LoRA HRW affinity,sticky slots and all-full fallback

Original explicit-score discrete allocator and independent feasible-subset
oracle. Synthetic,independently_reproduced=false.
No BLAKE3bitparity,Rust/Dynamo,controller,request routing,
adapter transfer/load,residency,inference or GPU reproduction.

```powershell
python -m experiments.lora_hrw_sticky_selection.run --check
python -m unittest discover -s tests -p test_lora_hrw_sticky_selection.py -q
```

`--measure` deliberately replaces original CPU precomputed-score top4
heap/fullsort samples. No hashing/slot/sticky policy included.
32-worker heap fixture is slower;not serving speed evidence.

Dynamopin`2cac4d767f49b5b000c0f354fe369554744bba2b`;
inspected HRW/interface files Apache-2.0.
Native name/workerID/DPBLAKE3 encoding recorded but not executed.
Explicit scoreties,priorretention,partialunderfill,
allfullbasicHRWexception and missingusageeligibility modeled.
LatestNov14treeCSSnotnewHRW;Nov6extraction/Nov11refactor not invention.
No upstream implementation copied.

StaticFigure-tieroriginal60-solidticketcabinet:
base,fourfeet,threetrays,sixrails,21cards,21tabs,fourfasteners.
Reusedoriginalgeometryasadapter-homemetaphor;
no browser inspection performed after prior approval refusal.

- [中文](../../docs/components/lora-hrw-sticky-slot-fallback-placement-selection-faq.md)
- [English](../../docs/components/lora-hrw-sticky-slot-fallback-placement-selection-faq.en.md)
- [H5](../../docs/reports/lora-hrw-sticky-slot-fallback-placement-sizer.html)
- [7 cards](../../docs/reports/lora-hrw-sticky-slot-fallback-placement-selection-cards.html)
- [Report](report.json),[CPU records](latency.json)
