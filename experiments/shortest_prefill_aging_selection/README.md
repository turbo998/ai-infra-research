# Round161:SPFaging clock and page reservation

Original bounded rational clock/rank/page oracle,not native scheduler or wall-time fairness.

```powershell
python -m experiments.shortest_prefill_aging_selection.run --check
python -m unittest discover -s tests -p test_shortest_prefill_aging_selection.py -q
```

Common-clock terms cancel once all requests arrived.Aging favorsoldbacklogagainstnewarrivals;doesnotcontinuallyreorderafixedpool.Nativefactor0remainsdefault.Originalexplicitfactor1/2,deferral-firstsort,prefixmaxnot sum,page-roundedreservationandextend-onlyclockaretested.

`--measure`replacesoriginalCPUsortsamples;ordinarygenerationretainssealeddata.Setup,nativecache/prefetch/admission,GPUsandqueuewaitsexcluded.[中文FAQ](../../docs/components/shortest-prefill-aging-clock-page-reservation-selection-faq.md) / [EnglishFAQ](../../docs/components/shortest-prefill-aging-clock-page-reservation-selection-faq.en.md).No browservisualacceptance.
