# io_uring credits, wraparound and ordinary-read buffer lifetime

Original sequential single-producer SQ/CQ model with uint32 wrap, request-ID routing, buffer ownership and explicit negative/short/full read results. A separate deque/identity oracle checks160deterministic events. Ordinary buffer lifetime is distinct from consumed SQ credit.

```powershell
python -m experiments.uring_ring_lifetime_selection.run --measure
python -m experiments.uring_ring_lifetime_selection.run --check
python -m unittest discover -s tests -p test_uring_ring_lifetime_selection.py -q
```

No Linux, atomic execution, SQPOLL, multishot/F_MORE, skip-success, zero-copy-send notification, cancellation or native CQ overflow. Hard CQ capacity and never-reused exact-integer IDs are original policies. CPU records only compare copied-list cursor versus shifting; not io_uring/epoll/read/IOPS. [English FAQ](../../docs/components/io-uring-ring-credits-wraparound-buffer-lifetime-selection-faq.en.md) / [中文 FAQ](../../docs/components/io-uring-ring-credits-wraparound-buffer-lifetime-selection-faq.md).
