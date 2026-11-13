# Round 101: CTC prefix decoding and streaming state

Original pure-acoustic references: enumerate paths, log-space prefix
blank/nonblank recurrence, and an independent fixed-target forward
automaton. Synthetic inputs do not establish ASR quality or calibrated
confidence. Local CPU timings compare these original exact functions,
not native pyctcdecode/NeMo or model inference.

```powershell
python -m experiments.ctc_prefix_decoding_selection.run
python -m experiments.ctc_prefix_decoding_selection.run --check
python -m unittest tests.test_ctc_prefix_decoding_selection
```

`latency.json` seals two warmups and seven alternating repeats on four
frame tiers. Normal rendering and checking never remeasure. Explicit
`--measure` replaces samples; do so only for a newly identified measurement
run and update corresponding timing prose. `--check --measure` is invalid.

See the six-section [English FAQ](../../docs/components/ctc-prefix-beam-blank-streaming-state-selection-faq.en.md)
and [中文 FAQ](../../docs/components/ctc-prefix-beam-blank-streaming-state-selection-faq.md),
[H5](../../docs/reports/ctc-prefix-beam-blank-streaming-state-sizer.html),
[seven cards](../../docs/reports/ctc-prefix-beam-blank-streaming-state-selection-cards.html)
and [SVG](../../assets/svg/ctc-prefix-beam-blank-streaming-state-selection-faq-summary.svg).

Pinned sources are linked in the FAQs. Upstream rights remain with their
holders; no implementation or model weights were copied. No LM, hotword,
BPE, timestamp or production finalization parity is claimed.
