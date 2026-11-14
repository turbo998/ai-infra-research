# Source-mode normalization before RGB/BGR order

Original six post-decode modes normalize to HWCthree-channel uint8.Alpha is discarded,not composited;palette indices resolve first;bilevel becomes0/255.No compressed codec,Pillow/NumPy/ONNXreference,Grayscale or colour-profile execution.

```powershell
python -m experiments.image_mode_channel_selection.run --measure
python -m experiments.image_mode_channel_selection.run --check
python -m unittest discover -s tests -p test_image_mode_channel_selection.py -q
```

Twelve original mode/order cases expose ten unsafe raw-path misses.Toy consumer scores are not model quality.CPU timing copies pre-normalized RGBtoBGR,not native decode/view or GPUupload.Native contiguity/peak/encoded-input immutability remain unknown.

[English FAQ](../../docs/components/onnx-image-decoder-source-mode-rgb-bgr-contract-selection-faq.en.md) / [中文](../../docs/components/onnx-image-decoder-source-mode-rgb-bgr-contract-selection-faq.md).Original60-solid figure;visual browser inspection absent.
