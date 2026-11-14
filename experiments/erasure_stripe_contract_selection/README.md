# Round133: checkpoint erasure stripes and failure domains

Original systematic Vandermonde GF256 polynomial0x11b,
matrix-inversion known-erasure decoder and independent Lagrange encoder.
Synthetic,independently_reproduced=false.
Not native Ceph/ISA-L byte parity,FastEC,network,disk or durability.

```powershell
python -m experiments.erasure_stripe_contract_selection.run --check
python -m unittest discover -s tests -p test_erasure_stripe_contract_selection.py -q
```

`--measure` deliberately replaces original parity-only CPU records.
Precomputed coefficient lookup versus bitwise field multiplication;
generator/table setup excluded. No native SIMD/rebuild benchmark.

Cephpin`ec322a5e95ebcb46dec7d6a9d7813a34543c64ad`;
linked documentation CC-BY-SA-3.0 perCOPYINGdoc/*.
Original model and factual summary,no copied codec/document text.
Latest tree releases completed push buffers,not new erasure theory.

Byte-packed codec and separate classic full-stripe budget are different
explicit representations. Native FastEC padding elimination not modeled.
Known erasures require k known-good identified survivors.
Silent corruption can produce wrong output;external expected checksum
fixture is not a complete integrity/commit protocol.

StaticFigure-tieroriginal60-solidticketcabinet:
base,fourfeet,threetrays,sixrails,21cards,21tabs,fourfasteners.
Reusedoriginalgeometryasshard-ticketmetaphor;
no browser inspection performed after prior approval refusal.

- [中文](../../docs/components/checkpoint-erasure-stripe-gf256-failure-domain-selection-faq.md)
- [English](../../docs/components/checkpoint-erasure-stripe-gf256-failure-domain-selection-faq.en.md)
- [H5](../../docs/reports/checkpoint-erasure-stripe-gf256-failure-domain-sizer.html)
- [7 cards](../../docs/reports/checkpoint-erasure-stripe-gf256-failure-domain-selection-cards.html)
- [Report](report.json),[CPU records](latency.json)
