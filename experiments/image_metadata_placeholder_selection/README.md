# Original ordered image metadata / placeholder planner

Round166 / daytime43 studies the actual2026-11-14vLLMmetadata-only Kimi-K3 update at
[`24fe0a99f1062f7351852804f2cd38668f7423bd`](https://github.com/vllm-project/vllm/commit/24fe0a99f1062f7351852804f2cd38668f7423bd).
Read the pinned resize/field/placeholder implementation,dictionary admission and both model embedding paths.
Original bounded geometry/occurrence model;no native implementation mirror or native model execution.

[中文FAQ](../../docs/components/kimi-k3-metadata-only-image-original-dimensions-placeholder-selection-faq.md) ·
[EnglishFAQ](../../docs/components/kimi-k3-metadata-only-image-original-dimensions-placeholder-selection-faq.en.md) ·
[H5](../../docs/reports/kimi-k3-metadata-only-image-original-dimensions-placeholder-sizer.html) ·
[Seven cards](../../docs/reports/kimi-k3-metadata-only-image-original-dimensions-placeholder-selection-cards.html) ·
[SVG](../../assets/svg/kimi-k3-metadata-only-image-original-dimensions-placeholder-selection-faq-summary.svg).

Default original profile:65×33/9×7/65×33,patch8/merge2/side16/budget256.
Counts15/1/15,total31;exclusive ranges[0,15),[15,16),[16,31).
Original headers and all occurrences remain;intentional padded headers miss3descriptors,
size-only collapse loses one occurrence and15pads.These are counterfactual proposals,not old native failures.
Metadata-only admission is not embedding delivery or model readiness.

`run.py` validates explicit original bounds and merge-aligned profiles;native valid domains are broader.
Declared embedding shapes verify row sums only,not tensors/hidden width/dtype/device.
`report.json` is synthetic/independently_reproduced=false.
`latency.json` measures original CPUgeometry memo versus direct computation with identical ordered outputs;
cache is per-call/configuration,native parser does not gain this memo from the studied patch.
Compact descriptor bytes and declared uncompressed RGBbudgets are not native traffic.
Native tokenizer IDs/encoder/EC/GPU/transport and browser acceptance are unknown.

```powershell
python -m unittest discover -s tests -p test_image_metadata_placeholder_selection.py -q
python -m experiments.image_metadata_placeholder_selection.run --check
# Explicitly replace original CPUevidence:
python -m experiments.image_metadata_placeholder_selection.run --measure
```

Eight focused tests,64seeded profiles plus default/195Nodeoutcomes.
Seven900×1125strict-CSPcards have no network/scripts;H5uses local inline JS.
AI-assisted original documentation and code;Apache-2.0source rights retained.
