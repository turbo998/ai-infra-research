# First-write placement and page-index weighted interleave

Original ideal anonymous-page placement model: local serial/future-owner first writes, explicit page-index interleave, weighted intervals, sticky existing ownership and fixed nonempty allowed-mask intersection. Independent expanded-cycle oracle covers30weight sets×128offsets.

```powershell
python -m experiments.numa_page_policy_selection.run --measure
python -m experiments.numa_page_policy_selection.run --check
python -m unittest discover -s tests -p test_numa_page_policy_selection.py -q
```

No Linux execution, allocator fallback, cpuset rebind, THP, shared page cache, read-zero-page fault, migration or auto-bandwidth weight derivation. Local first-write uses writer nodes; selected interleave nodes are not a bind restriction. CPU timings only compare original target lookups with preprocessing excluded, not memory access. [English FAQ](../../docs/components/numa-first-write-weighted-interleave-page-placement-selection-faq.en.md) / [中文 FAQ](../../docs/components/numa-first-write-weighted-interleave-page-placement-selection-faq.md).
