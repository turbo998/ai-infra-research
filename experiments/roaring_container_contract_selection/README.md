# Roaring array/bitset/run container payloads

Original unsigned32bit identity grouping, selected little-endian container payload encoder/decoder, sorted intersection and independent interval oracle. The format is a primary-source contract; the strict run-saving policy is original and not native `runOptimize`. Whole-file cookies/headers and native memory are excluded.

```powershell
python -m experiments.roaring_container_contract_selection.run --measure
python -m experiments.roaring_container_contract_selection.run --check
python -m unittest discover -s tests -p test_roaring_container_contract_selection.py -q
```

Original CPU samples compare prebuilt C-backed bigint intersection cardinality with Python sorted-intersection list materialization—unequal work, not native Roaring performance. Roaring64's new query interfaces are source-tracked but unexecuted. See [English FAQ](../../docs/components/roaring-array-bitset-run-container-payload-selection-faq.en.md) / [中文 FAQ](../../docs/components/roaring-array-bitset-run-container-payload-selection-faq.md).
