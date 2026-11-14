# Byte-preserving STRING tensor contract

Original typed flat text/hex records and bounded protobuf-compatible wire subset.Unicode encodes to UTF-8;opaque bytes,terminal NULs and empty fields remain unchanged.Not native ONNX/NumPy/protobuf execution or a general TensorProto parser.

```powershell
python -m experiments.string_tensor_bytes_selection.run --measure
python -m experiments.string_tensor_bytes_selection.run --check
python -m unittest discover -s tests -p test_string_tensor_bytes_selection.py -q
```

Naive trailing-NUL trim is an original mistake baseline,not complete historical NumPy reproduction.CPU timing assembles prebuilt fragments only;native serialization/network unknown.Scalar and zero-dimensional counts are checked;invalid UTF-8 inspection never silently changes bytes.

[English FAQ](../../docs/components/onnx-string-tensor-utf8-opaque-bytes-nul-contract-selection-faq.en.md) / [中文](../../docs/components/onnx-string-tensor-utf8-opaque-bytes-nul-contract-selection-faq.md).Original60-solid figure;visual browser inspection absent.
