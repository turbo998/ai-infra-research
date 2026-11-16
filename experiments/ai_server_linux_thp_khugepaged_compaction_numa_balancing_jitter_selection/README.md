# Round 311: AI Server Linux `THP` (`khugepaged` Direct Compaction) & `kernel.numa_balancing` `TLB Shootdown` Jitter Selection

Deterministic evaluator comparing 4 Linux kernel memory management configurations across 64-node (`512-GPU`) synchronous training on CPU P99 pause jitter (`ms`), step P99 slowdown (`%`), and cluster `MFU` (`%`).
