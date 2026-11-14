# Round134: Bloom negative gates and deletion provenance

Original explicit bit-coordinate model,exact finite coordinate-pair
universe and independent required-set/occupied-set oracle.
Synthetic,independently_reproduced=false.
No native Murmur/Redis,FPR,network,storage/database reproduction.

```powershell
python -m experiments.bloom_negative_gate_selection.run --check
python -m unittest discover -s tests -p test_bloom_negative_gate_selection.py -q
```

`--measure` deliberately replaces original CPU two-probe membership
samples. Packed bytearray versus bool-list,coordinates precomputed.
Packed is slower here;no native latency or allocator-footprint claim.

RedisBloompin`393f33d3fa9da26dd8be36f41cbca91655bdf8e9`.
sb.c offers RSALv2/SSPLv1/AGPLv3;deps/bloom has BSD header.
Original model,no upstream implementation copied.
Nov12installerretrychange not new Bloom theory.
Native chain p/2 then half targets,all-link precheck and explicit
non-scaling full behavior read;native rounding/hash not emulated.
Ideal sizing and finite layer bounds explicitly separate from measuredFPR.

StaticFigure-tieroriginal60-solidticketcabinet:
base,fourfeet,threetrays,sixrails,21cards,21tabs,fourfasteners.
Reusedoriginalgeometryasmembershipmetaphor;
no browser inspection performed after prior approval refusal.

- [中文](../../docs/components/bloom-negative-gate-layer-budget-safe-deletion-selection-faq.md)
- [English](../../docs/components/bloom-negative-gate-layer-budget-safe-deletion-selection-faq.en.md)
- [H5](../../docs/reports/bloom-negative-gate-layer-budget-safe-deletion-sizer.html)
- [7 cards](../../docs/reports/bloom-negative-gate-layer-budget-safe-deletion-selection-cards.html)
- [Report](report.json),[CPU records](latency.json)
