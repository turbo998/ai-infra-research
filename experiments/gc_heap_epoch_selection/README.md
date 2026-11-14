# Round164:startup GC freeze epochs

Original object-count lifecycle ledger plus isolated CPython measurements,not a vLLMworker.

```powershell
python -m experiments.gc_heap_epoch_selection.run --check
python -m unittest discover -s tests -p test_gc_heap_epoch_selection.py -q
```

Startup freeze skips original modeled static visits;final warmup unfreeze releases frozen cyclic garbage. Freeze does not stop refcount or automatic GC. Default capture exit restores prior state;opt-out leaves startup epoch until warmup. Actual probes run in child processes,never the unittest runner.

`--measure`replaces CPython latency/probe evidence;ordinary generation retains it. Setup/freeze/native vLLM/CUDA excluded. Entire child interpreter freeze counts are reported,not fabricated subset totals. [中文](../../docs/components/worker-startup-gc-freeze-epochs-refcount-warmup-selection-faq.md) / [English](../../docs/components/worker-startup-gc-freeze-epochs-refcount-warmup-selection-faq.en.md). Browser acceptance unknown.
