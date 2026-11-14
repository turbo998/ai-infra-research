# Logical KV dtype namespace

Original canonical JSONobject cache isolates logical formats despite shared physical backing;external writer witnesses distinguish identity,payload and exact scalar error.Not native SGLang keys,Mooncake,FP8casts or KVquality.

```powershell
python -m experiments.logical_kv_dtype_namespace_selection.run --measure
python -m experiments.logical_kv_dtype_namespace_selection.run --check
python -m unittest discover -s tests -p test_logical_kv_dtype_namespace_selection.py -q
```

Cold typed misses do not fall back to untyped pages.Native API requires resolved torch.dtype;UIstrings are original metadata.Layout/rank/tenant compatibility remains separate.CPUtimer compares identical original JSONkeys,not native digest/storage latency.

[English FAQ](../../docs/components/hicache-logical-kv-dtype-storage-namespace-selection-faq.en.md) / [中文](../../docs/components/hicache-logical-kv-dtype-storage-namespace-selection-faq.md).Original60-solid figure;visual browser inspection absent.
