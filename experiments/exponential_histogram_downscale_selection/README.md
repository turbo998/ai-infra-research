# Round 113: exponential histogram scale and boundary contracts

Original exact nonnegative-rational model; scales -10..0, zero threshold
zero, immutable counts. No OpenTelemetry SDK/collector execution.

```powershell
python -m experiments.exponential_histogram_downscale_selection.run --check
python -m unittest discover -s tests -p test_exponential_histogram_downscale_selection.py -q
```

`--measure` replaces original CPU timing deliberately; normal checks do
not remeasure. Eight precomputed states merge to the same complete
coarse histogram as raw replay. Initial histograms, native circular
buffers, concurrency, encoding and transport are excluded.

The ten-observation fixture fits span eight into three slots at scale
-2. Floor preserves value containment; signed truncation preserves the
same count but misplaces three observations. Exact p99=32 is bounded by
(16,256], not recovered exactly from counts. Empty quantiles/upscaling/
infeasible one-slot fits fail explicitly without dropping observations.

Figure-tier visual concept: a card sorter combines observations into
coarser magazines. Reuse the repository's original 60-solid scene:
four feet, one base, three supported trays, six side rails, 21 cards,
21 card tabs and four corner fasteners. This is a static architecture
analogy, not a measured physical instrument or native layout.
Browser visual inspection was not performed.

The H5 estimates selected fresh power-of-two uint64 counter capacity,
not retained history, SDK totals or OTLP bytes. Optional negative-side
capacity arithmetic does not extend the original observation model to
negative values.

- [中文 FAQ](../../docs/components/exponential-histogram-signed-index-downscale-selection-faq.md)
- [English FAQ](../../docs/components/exponential-histogram-signed-index-downscale-selection-faq.en.md)
- [Seven cards](../../docs/reports/exponential-histogram-signed-index-downscale-selection-cards.html)
- [H5 budget](../../docs/reports/exponential-histogram-signed-index-downscale-sizer.html)
- [Report](report.json) and [CPU samples](latency.json)

Pinned OpenTelemetry Go `63c5f44275f152fa0a9629463db43474fe7214e2`,
November 9, 2026 (Apache-2.0). October 30 circular-storage refactor and
November 9 deduplication are recent; histogram mathematics is longstanding.
