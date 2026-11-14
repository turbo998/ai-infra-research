# WAL reader snapshots and checkpoint completion

Original fixed-page version ledger with committed end marks, conservative safe backfill, sticky reader snapshots, hidden uncommitted tails and explicit reset gates. Independent immutable transaction-state oracle plus a separate benign sequential native SQLite fixture.

```powershell
python -m experiments.wal_snapshot_checkpoint_selection.run --measure
python -m experiments.wal_snapshot_checkpoint_selection.run --check
python -m unittest discover -s tests -p test_wal_snapshot_checkpoint_selection.py -q
```

`native_oracle.json` records actual same-thread synthetic SQLite operations, not throughput, crashes or concurrency races. Local3.49.1patch/backport status is not certified against the documented WAL-reset fixes. No host dependency is changed. Native SQL rows are not model physical pages. Temporary DBs are scoped here and cleaned after closing connections. The model omits native read-mark slots, binary salts/checksums, locks and fsync. [English FAQ](../../docs/components/sqlite-wal-reader-endmark-checkpoint-visibility-selection-faq.en.md) / [中文 FAQ](../../docs/components/sqlite-wal-reader-endmark-checkpoint-visibility-selection-faq.md).
