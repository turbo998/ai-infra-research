# Original XCDbalanced-tail ownership / one-round search

Round170/daytime47 reads actualNov14vLLM
[`2aa7eb3ea5cea8c99f484ca726be2f4d9c0ba418`](https://github.com/vllm-project/vllm/commit/2aa7eb3ea5cea8c99f484ca726be2f4d9c0ba418).
Scope:MiniMax sparse-decode tail placement,not all indexer changes or native FlyDSLexecution.

[中文FAQ](../../docs/components/rocm-minimax-xcd-balanced-tail-one-round-placement-selection-faq.md) ·
[EnglishFAQ](../../docs/components/rocm-minimax-xcd-balanced-tail-one-round-placement-selection-faq.en.md) ·
[H5](../../docs/reports/rocm-minimax-xcd-balanced-tail-one-round-placement-sizer.html) ·
[Seven cards](../../docs/reports/rocm-minimax-xcd-balanced-tail-one-round-placement-selection-cards.html) ·
[SVG](../../assets/svg/rocm-minimax-xcd-balanced-tail-one-round-placement-selection-faq-summary.svg).

Original10requests×8rows/8chiplets/256CUs/topk16/oneKVhead:
3balanced splits/240launches/0padding versus2request splits/256launches/96padding.
Whole groups keep locality;tail rows spread contiguously;request wins equal-split ties.
Explicit4request splits is not one-round capacity.
Requested XCD0/1uses plain-grid scope;occupancy2and general cost model not reproduced.
One-round max4is not native general max8.
The≤512grid-row subset does not reach native1024workspace guard.

`report.json`:synthetic/independently_reproduced=false.
`latency.json`:original CPUownership tally/enumeration,not GPU/NVFP4/MXFP8/bitwise/performance.
Source GPUtiming comments not archived as benchmark coordinates.
Unknown native workspace/performance stays null.

```powershell
python -m unittest discover -s tests -p test_xcd_tail_placement_selection.py -q
python -m experiments.xcd_tail_placement_selection.run --check
python -m experiments.xcd_tail_placement_selection.run --measure
```

Eight tests,64independent ownership/bijection cases,65profiles/130Nodeoutcomes.
Seven900×1125strict-CSPcards,separate inline-JSH5,framed original cabinet SVG.
Apache-2.0rights retained,no implementation mirror.AI-assisted;native/visual acceptance unknown.
