# Original signal-free managed-engine shutdown ledger

Round168/daytime45 reads actualNov14vLLM
[`ec94b6a7fc3b7141528113fbc89802829a914c25`](https://github.com/vllm-project/vllm/commit/ec94b6a7fc3b7141528113fbc89802829a914c25).
Frontend drain precedes launcher-only TERM;remaining budget still has a native5sminimum.
Group KILL cleans timeout and post-exit descendants.No actual processes or signals executed.

[中文FAQ](../../docs/components/managed-engine-launcher-sigterm-frontend-drain-budget-selection-faq.md) ·
[EnglishFAQ](../../docs/components/managed-engine-launcher-sigterm-frontend-drain-budget-selection-faq.en.md) ·
[H5](../../docs/reports/managed-engine-launcher-sigterm-frontend-drain-budget-sizer.html) ·
[Seven cards](../../docs/reports/managed-engine-launcher-sigterm-frontend-drain-budget-selection-cards.html) ·
[SVG](../../assets/svg/managed-engine-launcher-sigterm-frontend-drain-budget-selection-faq-summary.svg).

Original finish2s/6s,frontend6s,budget10s,launcher1safterTERM:
drain-first2completions/7send versus legacy group-first0completions.
Hung launcher:remaining4s→wait5s→11smodeled end,not hard10sdeadline.
Instant reap and finish-before-stop ties are declared assumptions,not native races.
No-PID/already-started cleanup/completion stays null;serve errors remain errors.
The observed-exit profile cannot also claim a live launcher PID.

`report.json` synthetic/independently_reproduced=false.
`latency.json` separately measures sorted-finish aggregate counting only;
sorting/events/OS/network/DP/signals/waiting excluded.
Native200mspolling and5sminimum are source constants,not local runtime measurements.
Native Rust shell tests were read,not run;no process kill operations belong to this experiment.

```powershell
python -m unittest discover -s tests -p test_managed_engine_drain_selection.py -q
python -m experiments.managed_engine_drain_selection.run --check
python -m experiments.managed_engine_drain_selection.run --measure
```

Eight tests,64finish-count oracle cases,65profiles/130Nodeoutcomes.
Seven900×1125strict-CSPcards,separate H5inline JS,framed original SVG.
Apache-2.0rights retained,no implementation mirror.AI-assisted;native/visual acceptance unknown.
