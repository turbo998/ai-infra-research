# Original MOSS-Audio profiling-budget integer ledger

Round169/daytime46 reads actualNov14vLLM
[`24d4c4949db5d6d8c3149eb3c8f2ddb1f2c0224d`](https://github.com/vllm-project/vllm/commit/24d4c4949db5d6d8c3149eb3c8f2ddb1f2c0224d).
Dummy length becomes mel_sr×30;audio budget uses ceil frames and three stride2ceil stages.
No waveform creation,NumPy,feature extraction,tokenizer,Torch or native profiler execution.

[中文FAQ](../../docs/components/moss-audio-profiling-dummy-sample-rate-budget-selection-faq.md) ·
[EnglishFAQ](../../docs/components/moss-audio-profiling-dummy-sample-rate-budget-selection-faq.en.md) ·
[H5](../../docs/reports/moss-audio-profiling-dummy-sample-rate-budget-sizer.html) ·
[Seven cards](../../docs/reports/moss-audio-profiling-dummy-sample-rate-budget-selection-cards.html) ·
[SVG](../../assets/svg/moss-audio-profiling-dummy-sample-rate-budget-selection-faq-summary.svg).

Original source-default sizing:480000samples/3000declared frames/375audio tokens.
Fixed16000samples:100declared frames/13tokens/362shortfall.
Ceil-frame witness is not native extracted-feature length for every configuration.
Default span377;markers26digits→403span,while audio token budget remains375.
Shared waveform versus per-item float32mel budgets are not allocator peak.

`report.json`:synthetic/independently_reproduced=false.
`latency.json`:measured original Python scalar integer formulas,not native Torch/GPU/STFT/profiling.
Overrides clamp/zero truthiness and seq_len-independent budget are explicit.
Native default-budget test read,not run;actual extraction/native peak remain null.

```powershell
python -m unittest discover -s tests -p test_audio_profile_budget_selection.py -q
python -m experiments.audio_profile_budget_selection.run --check
python -m experiments.audio_profile_budget_selection.run --measure
```

Eight tests,64random ceil identities,65profiles/130Nodeoutcomes.
Seven900×1125strict-CSPcards,separate inline-JSH5,framed original SVG.
Apache-2.0rights retained,no implementation mirror.AI-assisted;native/visual acceptance unknown.
