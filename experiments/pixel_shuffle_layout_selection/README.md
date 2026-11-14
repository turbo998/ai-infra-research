# PixelShuffle DCR/CRD layout contracts

Original exact-integer NCHW output-gather, independent input-scatter oracle, inverse/adjoint, direct SpaceToDepth opset gate and selected logical traffic budget. The pinned PyTorch TorchScript exporter selects CRD; equal output shapes do not prove equal permutations.

```powershell
python -m experiments.pixel_shuffle_layout_selection.run --measure
python -m experiments.pixel_shuffle_layout_selection.run --check
python -m unittest discover -s tests -p test_pixel_shuffle_layout_selection.py -q
```

SpaceToDepth28 mode/function-body capability is dated Aug31, not a November release. Original CPU paths compare retained mappings against repeated mapping work; no native ONNX/PyTorch, GPU, actual tensor strides, fusion or image-quality evaluation. [English FAQ](../../docs/components/pixel-shuffle-depth-space-dcr-crd-layout-selection-faq.en.md) / [中文 FAQ](../../docs/components/pixel-shuffle-depth-space-dcr-crd-layout-selection-faq.md).
