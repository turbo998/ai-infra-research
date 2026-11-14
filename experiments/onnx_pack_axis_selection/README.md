# ONNX Pack/Unpack axis,radix and word contracts

Original rank-two UINT8-code model with independent lanes,LSB-first bounded-reservoir packing,explicit logical count and exact radix groups. Python integers/JS BigInt preserve UINT32bit31and63bit group arithmetic. Independent explicit-bitstream oracle covers64cases.

```powershell
python -m experiments.onnx_pack_axis_selection.run --measure
python -m experiments.onnx_pack_axis_selection.run --check
python -m unittest discover -s tests -p test_onnx_pack_axis_selection.py -q
```

Source feature was added Nov13BJT,reviewed Nov14:opset29Pack/Unpack. Original strict validation is intentionally stronger than schema's unspecified invalid-input behavior/function diagnostics. UI rank2,dimension0..16,total128codes;standalone CPU timing exceeds those bounds. No numerical quantization,INT4tensor conversion,ONNX/reference/function/ORT/GPU execution or runtime support/peak-memory certification. Selected word bytes only.

[English FAQ](../../docs/components/onnx-pack-unpack-axis-radix-word-contract-selection-faq.en.md) / [中文](../../docs/components/onnx-pack-unpack-axis-radix-word-contract-selection-faq.md). Original60-solid figure;visual browser inspection not performed.
