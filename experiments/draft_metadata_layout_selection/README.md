# Draft metadata versus allocated KVlayout

Original plain per-layer planner separates metadata builders/selected workspace/attention-only CGfromall-group kernel constraints and retained KVviews.No vLLM/Torch/CUDAgraph execution.

```powershell
python -m experiments.draft_metadata_layout_selection.run --measure
python -m experiments.draft_metadata_layout_selection.run --check
python -m unittest discover -s tests -p test_draft_metadata_layout_selection.py -q
```

Mixed groups keep builders;provided kernel sizes are reused.Zero builders do not remove offsets/KVpayload.Unpadded byte ledger is not native packing/padding/peak.CPUdict/private256Bdummy constructor timing is not native workspace or GPUlatency.

[English FAQ](../../docs/components/vllm-context-kv-only-draft-metadata-layout-selection-faq.en.md) / [中文](../../docs/components/vllm-context-kv-only-draft-metadata-layout-selection-faq.md).Global graph readiness/native quality remain unknown.No visual browser acceptance.
