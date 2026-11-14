# SQLite64KiB field sentinels and leaf-space accounting

Original no-freeblock table-leaf header geometry, opaque-cell byte builder and independent interval-space oracle. File-header page-size1and leaf-content-offset0both represent65536in different fields; freeblock/cell-count zeros remain zero.

```powershell
python -m experiments.sqlite_page_sentinel_selection.run --measure
python -m experiments.sqlite_page_sentinel_selection.run --check
python -m unittest discover -s tests -p test_sqlite_page_sentinel_selection.py -q
```

The benign isolated native fixture reads its own actual4096/65536B SQLite empty-table pages. Local dbstat is unavailable and metrics remain null with explicit reason; today's upstream fix is source-verified, not executed as a newly patched binary. Opaque cells are not SQLite records. No freeblock chains, interior/overflow pages, tree/file integrity or native throughput certification. [English FAQ](../../docs/components/sqlite-64k-page-size-content-offset-sentinels-selection-faq.en.md) / [中文 FAQ](../../docs/components/sqlite-64k-page-size-content-offset-sentinels-selection-faq.md).
