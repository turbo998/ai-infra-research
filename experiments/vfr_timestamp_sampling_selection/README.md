# Round 107: VFR timestamp lookup, clip policies and duplicate output slots

Original integer-tick metadata model, synthetic inputs and separately measured
Python CPU lookup timings. `independently_reproduced=false`; no TorchCodec,
FFmpeg/NVDEC, actual decoded-frame count, model quality, GPU throughput or peak HBM.

- [中文 FAQ](../../docs/components/vfr-video-timestamp-clip-sampling-selection-faq.md)
- [English FAQ](../../docs/components/vfr-video-timestamp-clip-sampling-selection-faq.en.md)
- [H5 output budget](../../docs/reports/vfr-video-timestamp-clip-sampling-sizer.html)
- [Seven cards](../../docs/reports/vfr-video-timestamp-clip-sampling-selection-cards.html)
- [Original SVG](../../assets/svg/vfr-video-timestamp-clip-sampling-selection-faq-summary.svg)

```powershell
python -m experiments.vfr_timestamp_sampling_selection.run
python -m unittest tests.test_vfr_timestamp_sampling_selection -q
python -m experiments.vfr_timestamp_sampling_selection.run --check
```

`--measure` explicitly replaces sealed samples. Ordinary generation/checking
does not remeasure. Both CPU paths validate requests with PTS metadata already
available; the linear oracle constructs its interval tuple per call.
Raw noisy shared-host samples are retained. Scanning/index preparation, I/O,
codecs, seeks, conversion and tensor output are excluded.

## Evidence and limits

Unique strictly increasing PTS define half-open next-PTS hold intervals.
Average-FPS indexing disagrees on five of nine synthetic VFR requests but agrees
on all 200 CFR control ticks. Stream-origin shifts preserve exact identities.
Duplicate target lookup preserves every original output slot.
Clip wrap cycles the valid timestamp list; repeat_last need not select the
video's final frame. Invalid times and duplicate/nonmonotonic PTS are rejected.

Unsorted synthetic labels illustrate presentation sorting; they do not model
actual packet/frame correspondence or B-frame decoding.
The target count is not total codec-decoded frames because seek pre-roll and
codec dependencies are unmodeled. H5 payloads exclude pitch, surfaces, conversion,
allocator and peak HBM; a unique-target tensor is hypothetical, not guaranteed savings.

TorchCodec pin `be5a80b79e19d928125a660f3c8f067af1fe6866` is the November 9,
2026 stable low-level API graduation, not a new VFR algorithm.
BSD-3-Clause rights remain with upstream holders. No source implementation,
encoded media, datasets or model weights are copied.
