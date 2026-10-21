# FlashInfer Cake All-Gather Matmul (`SM100a`/`SM103a`), Cake MSA/VSA Hopper (`SM90a`), and Blackwell `tcgen05` TMEM Lane-Quarter & CTA-Pair Stage-Drain Verification

Deterministic Python 3.12 standard-library verification for **Phase 46 (`priority: 55`)** Topic 1 (`q-phase46-flashinfer-cake-allgather-hopper-msa-vsa-tmem-race`).

## First-Hand Upstream Commits
- `flashinfer@d9695940c53e0563cb0855f93c0955660c143d9f` (`#6865`): SM100a ws2 CTA-pair `_cg2` routing (`1024x2048, K=8192`, `1.027x–1.065x`) and ws4 `512x2560, K=8192` 16-CTA-per-peer SM-push launch (`48` vs `24` push CTAs, `1.066x–1.086x`).
- `flashinfer@9c68e44fd2a7664d53d49168ed616aa8c06ef1f7` (`#6761`): SM103a ws8 `1024x1280, K=8192` 16-CTA-per-peer SM-push launch (`112` vs `56` push CTAs, `1.012x–1.034x`).
- `flashinfer@8a93c9b833761c75939f176a58268ad77ac304f1` (`#6252`) & `flashinfer@7f6f119d8534441430934e798ba056f052d1ba20` (`#6725`): 89-program Cake MSA Hopper (`sm_90a`) backend (`3.237x` on Top-K `nvp_forced_h4_tiles512_tq256`) and Blackwell persistent heavy-first GQA-folded union prefill (`1.825x` GB300 flat union, `3.300x` GB300 paged `first_topk`).
- `flashinfer@ddf080d8a56c87f60a25ee9df750e373c4bcbe92` (`#6877`): 17-program Cake VSA Hopper (`sm_90a`) backend with 4-TMA-box `V^T` load and ragged queue (`1.2524x` at `h1-m512-n8192-k48`).
- `flashinfer@c8be875c7c403c4eac664f034b098157d9dc0642` (`#6778`), `flashinfer@e4e7510f51fe9f058390ad84c66178805c8b2f45` (`#6767`), `flashinfer@653cce0a6e903a042d374cd509002bd82ae6d3a3` (`#6820`), & `flashinfer@2b3cd4f2f990954ea9a196c352dadae25fd3ff8c` (`#6763`): Blackwell PTX `tcgen05` warpgroup TMEM lane-quarter base `(tid & 96) << 16`, warp-uniform `(consumer_warp * 32) << 16` epilogue addressing, and `cta_group::2` `multicast::cluster` `producer_tail` `mma_done` stage-release drain across 50 kernels.

## Usage
```bash
python experiments/flashinfer_cake_allgather_hopper_msa_vsa_tmem_race/run.py --check
```
