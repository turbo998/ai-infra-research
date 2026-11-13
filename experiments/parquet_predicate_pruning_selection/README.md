# Round 112: conservative Parquet predicate pruning

Original, standard-library integer/null SQL truth-set interpreter and
independent concrete row evaluator. No DuckDB or Parquet execution.

```powershell
python -m experiments.parquet_predicate_pruning_selection.run --check
python -m unittest discover -s tests -p test_parquet_predicate_pruning_selection.py -q
```

`--measure` deliberately replaces original CPU timing; do not use it to
chase favorable results. `--check` verifies sealed outputs without
remeasuring. Two warmups, seven alternating repeats, three queries per
sample; precomputed metadata, no decoding/storage/native-engine timing.

Four groups and `WHERE NOT(x>9)` yield six matches with only groups 0/3
read. Boolean NOT of may-match loses three matches. Missing metadata
keeps extra groups; constant bounds do not prove no nulls; correlation
loss admits false positives. Soundness requires trusted conservative
metadata, not corrupted bounds.

The H5's uniform compressed chunks, metadata allowance and selected
group counts are synthetic inputs, not predicted IO/selectivity.
Actual read bytes, wire bytes, native latency and peak memory are null.

- [中文 FAQ](../../docs/components/parquet-sql-null-row-group-pruning-selection-faq.md)
- [English FAQ](../../docs/components/parquet-sql-null-row-group-pruning-selection-faq.en.md)
- [Seven cards](../../docs/reports/parquet-sql-null-row-group-pruning-selection-cards.html)
- [H5 budget](../../docs/reports/parquet-sql-null-row-group-pruning-sizer.html)
- [Sealed report](report.json) and [original CPU samples](latency.json)

Pinned DuckDB source `77e8a4971900bfea7161bd48e027fea589cb041e`
(2026-11-11), MIT. November 5 scratch-buffer changes establish chronology,
not new pruning semantics or measured upstream performance. Original
integer analogue does not claim constant-string/native optimizer parity.
