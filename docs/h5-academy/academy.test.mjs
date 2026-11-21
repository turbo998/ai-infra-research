import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { Script } from "node:vm";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import M from "./academy-math.cjs";
import { validateContentSources, b } from "./content.mjs";

const directory = dirname(fileURLToPath(import.meta.url));
const near = (actual, expected, epsilon = 1e-7) => assert.ok(Math.abs(actual - expected) < epsilon, `${actual} != ${expected}`);
const sizing = { model: "llama70", gpu: "h200", weightBytes: 1, kvBytes: 2, batch: 32, context: 8192, prefillTokens: 8192, prefillBatch: 1, safe: 0.85, runtime: 3, overhead: 0.05, replication: 1, mbu: 0.6, mfu: 0.4, tpot: 20, ttft: 800, reserve: 100, mtp: false };
const capacity = { qps: 10, headroom: 1.3, prompt: 2048, output: 512, ttft: 800, tpot: 20, inputCapacity: 20000, outputCapacity: 1500, batch: 128, utilization: 0.7 };
const transfer = { model: "llama70", context: 8192, kvBytes: 1, cache: 0, rate: 400, efficiency: 0.8, rails: 1, wireCopies: 1, customKVBytes: false, setup: 0.2, overlap: 10, qps: 10 };

test("TTFT and E2EL count the first output token exactly once", () => {
  const r = M.lifecycle({ queue: 120, prefill: 300, delivery: 30, output: 128, tpot: 20 });
  assert.equal(r.ttft, 450);
  assert.equal(r.e2e, 2990);
  assert.equal(r.userTPS, 50);
  const single = M.lifecycle({ queue: 120, prefill: 300, delivery: 30, output: 1, tpot: 20 });
  assert.equal(single.e2e, 450);
  assert.equal(single.userTPS, null);
  assert.equal(single.measuredTPOT, null);
});
const traceOptions = { requests: M.traceSamples, windowMs: 1000, ttftLimit: 300, tpotLimit: 35, gateMode: "max", boundary: "client", requireQuality: true };
test("visible traces preserve failure denominators and gate bursty streams by the agreed latency metric", () => {
  const r = M.traceMetrics(traceOptions);
  assert.equal(r.visibleOutputTPS, 18);
  assert.equal(r.goodOutputTPS, 5);
  assert.equal(r.goodRequests, 2);
  assert.equal(r.completedRPS, 7);
  assert.equal(r.admittedRPS, 6);
  assert.equal(r.statuses.failed, 1);
  assert.equal(r.statuses.rejected, 1);
  assert.equal(r.itlSamples, 12);
  assert.equal(r.p99ITL, 60);
  assert.equal(r.rows.find((row) => row.id === "burst").tpot, 20);
  assert.equal(M.traceMetrics({ ...traceOptions, gateMode: "mean" }).goodOutputTPS, 9);
  assert.equal(M.traceMetrics({ ...traceOptions, gateMode: "mean", requireQuality: false }).goodOutputTPS, 12);
  const cancelled = M.traceMetrics({ ...traceOptions, requests: [{ ...M.traceSamples[4], id: "cancelled", status: "cancelled" }] });
  assert.equal(cancelled.statuses.cancelled, 1);
  assert.equal(cancelled.visibleOutputTPS, 2);
  assert.equal(cancelled.goodOutputTPS, 0);
});
test("one-token and zero-resolution chunk intervals keep mathematical N/A separate from measured zero", () => {
  const single = M.traceMetrics({ ...traceOptions, requests: [M.traceSamples[3]] });
  assert.equal(single.rows[0].tpot, null);
  assert.equal(single.rows[0].userTPS, null);
  assert.equal(single.p99ITL, null);
  const chunk = M.traceMetrics({ ...traceOptions, requests: [{ ...M.traceSamples[0], visible_token_ms: [100, 100, 100] }] });
  assert.equal(chunk.rows[0].tpot, 0);
  assert.equal(chunk.rows[0].userTPS, null);
  assert.equal(chunk.p99ITL, 0);
});
test("trace occupancy is an exact time-area calculation and distinguishes client from admitted boundaries", () => {
  const r = M.traceMetrics(traceOptions);
  near(r.meanActive, 1.475);
  near(r.littleProduct, r.meanActive);
  assert.equal(r.closedWindow, true);
  const admitted = M.traceMetrics({ ...traceOptions, boundary: "admitted" });
  near(admitted.meanActive, 1.35);
  near(admitted.meanResidenceMs, 225);
  assert.equal(admitted.lambda, 6);
  assert.equal(admitted.completedRPS, 7);
  const clipped = M.traceMetrics({ ...traceOptions, windowMs: 200 });
  near(clipped.meanActive, 4.695);
  near(clipped.littleProduct, 1.345);
  assert.equal(clipped.closedWindow, false);
});
test("missing admission and unknown quality are not quietly replaced with zeros or passing annotations", () => {
  const unknown = [{ ...M.traceSamples[3], admitted_ms: null, quality_pass: null }];
  const r = M.traceMetrics({ ...traceOptions, requests: unknown });
  assert.equal(r.unknownAdmission, 1);
  assert.equal(r.goodOutputTPS, 0);
  assert.equal(M.traceMetrics({ ...traceOptions, requests: unknown, requireQuality: false }).goodOutputTPS, 1);
  assert.throws(() => M.traceMetrics({ ...traceOptions, requests: unknown, boundary: "admitted" }), RangeError);
});
test("trace schema rejects unsorted/mismatched timing, duplicate IDs and extra nonnumeric payload", () => {
  assert.throws(() => M.traceMetrics({ ...traceOptions, requests: [{ ...M.traceSamples[0], visible_token_ms: [100, 90] }] }), RangeError);
  assert.throws(() => M.traceMetrics({ ...traceOptions, requests: [{ ...M.traceSamples[0], ended_ms: 100 }] }), RangeError);
  assert.throws(() => M.traceMetrics({ ...traceOptions, requests: [M.traceSamples[0], M.traceSamples[0]] }), RangeError);
  assert.throws(() => M.traceMetrics({ ...traceOptions, requests: [{ ...M.traceSamples[0], prompt_text: "not part of the numeric schema" }] }), RangeError);
  assert.throws(() => M.traceMetrics({ ...traceOptions, requests: [] }), RangeError);
  assert.throws(() => M.traceMetrics({ ...traceOptions, requests: [{ ...M.traceSamples[0], arrival_ms: 10000, admitted_ms: 10020, visible_token_ms: [10080], ended_ms: 10142 }] }), RangeError);
});
test("correlated output lengths and TPOT cannot be replaced by the product of their separate means", () => {
  const requests = [
    { ...M.traceSamples[0], id: "short", visible_token_ms: [20, 30], ended_ms: 32, admitted_ms: 0 },
    { ...M.traceSamples[0], id: "long", visible_token_ms: [20, 120, 220, 320, 420], ended_ms: 422, admitted_ms: 0 }
  ];
  const r = M.traceMetrics({ ...traceOptions, requests });
  const meanDecode = r.rows.reduce((sum, row) => sum + row.e2eVisible - row.ttft, 0) / 2;
  near(meanDecode, 205);
  near(r.meanRequestTPOT, 55);
  near(r.meanObservedITL, 82);
  assert.notEqual(meanDecode, (3.5 - 1) * r.meanRequestTPOT);
});
test("capacity preserves independent input/output scopes and Little's Law units", () => {
  const r = M.capacity(capacity);
  assert.equal(r.qps, 13);
  assert.equal(r.inputTPS, 26624);
  assert.equal(r.outputTPS, 6656);
  assert.equal(r.inputTPM, 1597440);
  assert.equal(r.outputTPM, 399360);
  near(r.residence, 11.02);
  near(r.active, 143.26);
  assert.deepEqual([r.inputInstances, r.outputInstances, r.batchInstances, r.instances], [2, 7, 2, 7]);
});
test("zero demand has no false provisioning and single-token residence has no decode interval", () => {
  const r = M.capacity({ ...capacity, qps: 0 });
  assert.equal(r.instances, 0);
  assert.equal(r.active, 0);
  const s = M.capacity({ ...capacity, output: 1 });
  near(s.residence, 0.8);
  assert.equal(s.decodeActive, 0);
});
test("DAU conversion divides by active business hours rather than all-day seconds", () => {
  near(M.businessQPS({ dau: 20000, requests: 10, hours: 10, peak: 3 }), 600000 / 36000);
  assert.throws(() => M.businessQPS({ dau: 1, requests: 1, hours: 0, peak: 1 }), RangeError);
});
test("mixed capacity computes per-class residence before averaging and exposes long-request concurrency dominance", () => {
  const p = { qps: 10, headroom: 1.3, shortShare: 0.5, ragShare: 0.4, inputCapacity: 20000, outputCapacity: 1500, batch: 128, utilization: 0.7 };
  const r = M.mixtureCapacity(p);
  near(r.inputTPS, 33612.8);
  near(r.outputTPS, 8403.2);
  near(r.meanResidence, 28.7215);
  near(r.active, 373.3795);
  near(r.incorrectShortcutW, 17.7231);
  assert.ok(r.classes[2].active / r.active > 0.71);
  assert.equal(r.instances, 9);
  assert.equal(M.mixtureCapacity({ ...p, qps: 0 }).instances, 0);
  assert.throws(() => M.mixtureCapacity({ ...p, shortShare: 0.8, ragShare: 0.4 }), RangeError);
});
test("fluid burst queues drain from their own capped backlog rather than the unbounded backlog", () => {
  const p = { baseQPS: 8, burstMultiplier: 3, burstSeconds: 30, serviceRPS: 10, queueBudgetMs: 300, horizonSeconds: 300 };
  const r = M.burstQueue(p);
  assert.equal(r.peakQueue, 420);
  assert.equal(r.peakWaitSeconds, 42);
  assert.equal(r.drainSeconds, 210);
  assert.equal(r.queueCap, 3);
  assert.equal(r.cappedDrainSeconds, 1.5);
  assert.equal(r.rejectedDuringBurst, 417);
  const drained = r.points.find((point) => point.time === 31.5);
  assert.equal(drained.capped, 0);
  assert.equal(drained.queue, 417);
  assert.equal(M.burstQueue({ ...p, baseQPS: 12 }).drainSeconds, null);
  assert.equal(M.burstQueue({ ...p, baseQPS: 0 }).peakQueue, 0);
  assert.throws(() => M.burstQueue({ ...p, serviceRPS: 0 }), RangeError);
});
test("PTU cache and output weights do not silently use a universal TPM conversion", () => {
  const p = { inputTPM: 200000, outputTPM: 20000, cache: 0.5, cachedWeight: 0, outputWeight: 8, capacity: 3400, minimum: 15, step: 5 };
  const r = M.ptu(p);
  assert.equal(r.normalized, 260000);
  near(r.raw, 260000 / 3400);
  assert.equal(r.provisioned, 80);
  assert.equal(M.ptu({ ...p, cache: 0 }).provisioned, 110);
  assert.equal(M.ptu({ ...p, inputTPM: 0, outputTPM: 0 }).provisioned, 0);
  assert.equal(M.ptu({ ...p, inputTPM: 1, outputTPM: 0 }).provisioned, 15);
});
test("PTU scale increment is anchored to the declared minimum, including non-aligned grids", () => {
  assert.equal(M.ptu({ inputTPM: 23, outputTPM: 0, cache: 0, cachedWeight: 0, outputWeight: 1, capacity: 1, minimum: 12, step: 5 }).provisioned, 27);
});
test("GPT-6 normalized categories include cache reads/writes exactly once and round by deployment", () => {
  const p = { model: "gpt62sol", deployment: "global-zone", shortInput: 100000, shortRead: 100000, shortWrite30: 20000, shortWrite12: 0, shortOutput: 20000, longInput: 10000, longRead: 10000, longWrite30: 2000, longWrite12: 0, longOutput: 2000 };
  const r = M.normalizedPTU(p);
  assert.equal(r.rawTokens, 264000);
  assert.equal(r.normalized, 271000);
  assert.equal(r.provisioned, 95);
  assert.equal(r.incorrectFreeReads, 90);
  assert.equal(r.readContribution, 6000);
  assert.equal(M.normalizedPTU({ ...p, deployment: "regional" }).provisioned, 100);
  assert.equal(M.normalizedPTU({ ...p, model: "gpt6sol" }).normalized, 277000);
  const zero = { ...p };
  for (const key of r.categories.map((category) => category.key)) zero[key] = 0;
  assert.equal(M.normalizedPTU(zero).provisioned, 0);
  assert.throws(() => M.normalizedPTU({ ...p, model: "unknown" }), RangeError);
  assert.throws(() => M.normalizedPTU({ ...p, shortRead: null }), RangeError);
});
test("GQA and MLA hand calculations use distinct cache representations", () => {
  assert.equal(M.kvPerToken(M.models.llama70, 2), 327680);
  assert.equal(M.kvPerToken(M.models.deepseek, 2), 70272);
  assert.equal(M.kvPerToken(M.models.deepseek, 1), 35136);
  assert.equal(M.kvPerToken(M.models.llama70, 2, "mha") / M.kvPerToken(M.models.llama70, 2), 8);
  assert.throws(() => M.kvPerToken(M.models.llama70, 2, "mla"), RangeError);
});
test("GPU sizing distinguishes nominal GB from GiB, includes runtime per GPU and full KV reads", () => {
  const r = M.sizing(sizing);
  near(r.weight, 74.13);
  near(r.kv, 85.89934592);
  near(r.kv * 1e9 / 2 ** 30, 80);
  assert.equal(r.memoryCards, 2);
  assert.equal(r.bandwidthCards, 3);
  assert.equal(r.computeCards, 5);
  assert.equal(r.candidate, 8);
  assert.ok(r.unionTPOT + r.kvReadTPOT <= sizing.tpot);
});
test("future output grows resident KV without inventing extra cold-prefill FLOPs", () => {
  const r = M.sizing(sizing);
  const longerResident = M.sizing({ ...sizing, context: 8704 });
  assert.ok(longerResident.kv > r.kv);
  assert.equal(longerResident.prefillFLOPs, r.prefillFLOPs);
  const shorterPrompt = M.sizing({ ...sizing, prefillTokens: 4096 });
  assert.equal(shorterPrompt.kv, r.kv);
  assert.ok(shorterPrompt.prefillFLOPs < r.prefillFLOPs);
  assert.throws(() => M.sizing({ ...sizing, prefillTokens: 8193 }), RangeError);
  assert.throws(() => M.sizing({ ...sizing, prefillBatch: 33 }), RangeError);
});
const rank = { model: "llama70", gpu: "h200", tp: 16, pp: 1, dp: 1, batch: 32, context: 8192, weightBytes: 1, kvBytes: 2, safe: 0.85, runtime: 3, overhead: 0.05, weightSkew: 1, mlaCache: "replicated", mlaLayerBytes: 1152, measuredMemory: false, customReplicaBatch: false, mtp: false };
test("GQA KV heads stop shrinking beyond their count and total TP cache replication is exposed", () => {
  const tp8 = M.rankMemory({ ...rank, tp: 8 });
  const tp16 = M.rankMemory(rank);
  assert.equal(tp8.kvHeadsPerRank, 1);
  assert.equal(tp16.kvHeadsPerRank, 1);
  near(tp16.stages[0].kvGB, tp8.stages[0].kvGB);
  near(tp16.kvReserveGB, tp8.kvReserveGB * 2);
  near(tp16.effectiveKVFactor, 2);
  assert.equal(tp16.cards, 16);
});
test("MLA reference replication and opaque measured bytes are not blindly divided by TP", () => {
  const replicated = M.rankMemory({ ...rank, model: "deepseek", tp: 8 });
  near(replicated.effectiveKVFactor, 8);
  const measured = M.rankMemory({ ...rank, model: "deepseek", tp: 8, mlaCache: "measured", mlaLayerBytes: 656, kvBytes: 1 });
  assert.equal(measured.bytesPerLayer, 656);
  near(measured.effectiveKVFactor, 8 * 656 / 576);
  assert.throws(() => M.rankMemory({ ...rank, model: "deepseek", mlaCache: "unknown" }), RangeError);
});
test("uneven PP accounts for stage layers and ordinary DP preserves replicated full weights", () => {
  const r = M.rankMemory({ ...rank, model: "deepseek", tp: 8, pp: 2, dp: 2, batch: 33 });
  assert.deepEqual(r.stages.map((s) => s.layers), [31, 30]);
  assert.equal(r.cards, 32);
  assert.equal(r.replicaBatch, 17);
  assert.equal(r.reservedRequestSlots, 34);
  near(r.weightReserveGB, 671 * 1.05 * 2);
  assert.ok(r.stages[0].usedGB > r.stages[1].usedGB);
});
test("actual device capacity and busiest replica override conditional memory fit, not runtime support", () => {
  const design = M.rankMemory({ ...rank, gpu: "b200", tp: 1 });
  const delivered = M.rankMemory({ ...rank, gpu: "b200dgx", tp: 1 });
  assert.equal(design.fits, true);
  assert.equal(delivered.fits, false);
  assert.equal(M.rankMemory({ ...rank, tp: 1, measuredMemory: true, deviceGB: 180 }).fits, false);
  const busy = M.rankMemory({ ...rank, dp: 8, customReplicaBatch: true, replicaBatch: 32 });
  assert.equal(busy.replicaBatch, 32);
  assert.equal(busy.reservedRequestSlots, 256);
  assert.throws(() => M.rankMemory({ ...rank, customReplicaBatch: true, replicaBatch: 33 }), RangeError);
  assert.throws(() => M.rankMemory({ ...rank, model: "llama8", tp: 64 }), RangeError);
});
test("MoE total weights stay resident while the batch expert union grows", () => {
  near(M.moeUnion(M.models.deepseek, 1), 37);
  assert.ok(M.moeUnion(M.models.deepseek, 32) > 37);
  assert.ok(M.moeUnion(M.models.deepseek, 32) < 671);
  near(M.moeUnion(M.models.deepseek, 4096), 671);
  const one = M.sizing({ ...sizing, model: "deepseek", batch: 1 });
  const many = M.sizing({ ...sizing, model: "deepseek", batch: 32 });
  near(one.weight, 671 * 1.05);
  assert.equal(many.weight, one.weight);
  assert.ok(many.unionTraffic > one.unionTraffic);
});
test("MTP is separately budgeted; quantized weight precision never changes KV precision", () => {
  const base = M.sizing({ ...sizing, model: "deepseek" });
  const mtp = M.sizing({ ...sizing, model: "deepseek", mtp: true });
  near(mtp.weight - base.weight, 14 * 1.05);
  const int4 = M.sizing({ ...sizing, weightBytes: 0.5 });
  near(int4.weight, M.sizing(sizing).weight / 2);
  assert.equal(int4.kv, M.sizing(sizing).kv);
});
test("sizing rejects impossible memory/TTFT reserves and unknown hardware rather than manufacturing success", () => {
  assert.throws(() => M.sizing({ ...sizing, runtime: 200 }), RangeError);
  assert.throws(() => M.sizing({ ...sizing, reserve: 800 }), RangeError);
  assert.throws(() => M.sizing({ ...sizing, gpu: "unknown" }), RangeError);
  assert.throws(() => M.sizing({ ...sizing, safe: 0 }), RangeError);
  assert.throws(() => M.sizing({ ...sizing, context: Infinity }), RangeError);
  assert.equal(M.sizing({ ...sizing, model: "qwen72", context: 65536 }).contextSupported, false);
});
test("GPU floor calculations satisfy their exact declared memory, bandwidth and cold-prefill budgets across presets", () => {
  for (const model of Object.keys(M.models)) for (const gpu of Object.keys(M.gpus)) {
    for (const context of [512, 8192, 32768]) for (const batch of [1, 16, 64]) for (const weightBytes of [0.5, 2]) {
      const p = { ...sizing, model, gpu, context, prefillTokens: Math.min(context,8192), batch, weightBytes };
      const r = M.sizing(p);
      assert.ok(r.weight+r.kv+p.runtime*r.memoryCards <= r.gpu.memory*p.safe*r.memoryCards+1e-7);
      assert.ok((r.unionTraffic+r.kv)/(r.candidate*r.gpu.bandwidth*p.mbu)*1000 <= p.tpot+1e-7);
      assert.ok(r.prefillMs+p.reserve <= p.ttft+1e-7);
    }
  }
});
test("B200 design and shipped DGX capacities remain distinct and L40S does not get NVLink", () => {
  const design = M.sizing({ ...sizing, gpu: "b200" });
  const shipped = M.sizing({ ...sizing, gpu: "b200dgx" });
  assert.equal(design.gpu.memory, 192);
  assert.equal(shipped.gpu.memory, 180);
  assert.equal(M.sizing({ ...sizing, gpu: "l40s" }).gpu.nvlink, 0);
});
test("PD converts bits to bytes, applies efficiency once and subtracts only verified overlap", () => {
  const r = M.transfer(transfer);
  assert.equal(r.bytes, 1342177280);
  assert.equal(r.bandwidth, 40e9);
  near(r.rawMs, 33.554432);
  near(r.exposedMs, 23.754432);
  near(r.utilization, 0.33554432);
  const two = M.transfer({ ...transfer, rails: 2 });
  near(two.rawMs, r.rawMs / 2);
  const hit = M.transfer({ ...transfer, cache: 1 });
  assert.equal(hit.bytes, 0);
  assert.equal(hit.exposedMs, 0.2);
});
test("PD exposed handoff never becomes negative and overload stays explicitly measurable", () => {
  assert.equal(M.transfer({ ...transfer, overlap: 1000 }).exposedMs, 0.2);
  assert.ok(M.transfer({ ...transfer, qps: 100 }).utilization > 1);
  assert.throws(() => M.transfer({ ...transfer, rate: 0 }), RangeError);
});
test("PD wire bytes stay distinct from unique logical state and accept confirmed opaque token geometry", () => {
  const one = M.transfer(transfer);
  const two = M.transfer({ ...transfer, wireCopies: 2 });
  assert.equal(two.logicalBytes, one.logicalBytes);
  assert.equal(two.bytes, one.bytes * 2);
  near(two.rawMs, one.rawMs * 2);
  const custom = M.transfer({ ...transfer, model: "deepseek", customKVBytes: true, bytesPerToken: 40016 });
  assert.equal(custom.bytesPerToken, 40016);
  assert.equal(custom.bytes, 40016 * 8192);
});
const pd = { ...transfer, outputTokens: 512, destinationCopies: 1, prefillPoolTPS: 200000, decodePoolTPS: 12000, decodeSlots: 256, meanTPOT: 20, capacityLimit: 0.7, firstVisible: "decode", queueP: 100, prefillBudget: 350, queueD: 50, firstBudget: 20, delivery: 30, ttftTarget: 800, itlTarget: 35, prefillReplicas: 1, decodeReplicas: 1, weightBytes: 1, weightOverhead: 0.05, compatible: true, memoryChecked: true, colocatedPasses: false, mtp: false };
test("PD pipeline takes the minimum pool/network/slot bound and evaluates engineering latency separately", () => {
  const r = M.pdBudget(pd);
  assert.equal(r.readiness, "trial-candidate");
  assert.equal(r.bottleneck.key, "D output");
  near(r.bottleneck.rps, 23.4375);
  assert.equal(r.candidateOutputTPS, 12000);
  near(r.ttftBudget, 573.754432);
  assert.equal(r.firstGapBudget, 20);
  near(r.weightsGB, 148.26);
  assert.equal(M.pdBudget({ ...pd, wireCopies: 4 }).readiness, "blocked-capacity");
  assert.equal(M.pdBudget({ ...pd, decodeSlots: 1 }).bottleneck.key, "D slots");
  assert.equal(M.pdBudget({ ...pd, compatible: false }).readiness, "blocked-layout");
  assert.equal(M.pdBudget({ ...pd, colocatedPasses: true }).readiness, "keep-baseline");
});
test("early P output can improve TTFT while violating the first streaming gap", () => {
  const r = M.pdBudget({ ...pd, firstVisible: "prefill" });
  assert.equal(r.ttftBudget, 500);
  near(r.firstGapBudget, 93.754432);
  assert.equal(r.readiness, "blocked-latency");
  assert.throws(() => M.pdBudget({ ...pd, firstVisible: "prefill", outputTokens: 1 }), RangeError);
  assert.throws(() => M.pdBudget({ ...pd, overlap: 351 }), RangeError);
});
test("D state replication and cached prefixes do not automatically multiply or erase wire traffic", () => {
  const one = M.pdBudget(pd);
  const replicated = M.pdBudget({ ...pd, destinationCopies: 2 });
  assert.equal(replicated.tx.bytes, one.tx.bytes);
  assert.equal(replicated.fullStateBytes, one.fullStateBytes * 2);
  const cached = M.pdBudget({ ...pd, cache: 1 });
  assert.equal(cached.tx.bytes, 0);
  assert.equal(cached.fullStateBytes, one.fullStateBytes);
  assert.ok(!cached.limits.some((limit) => limit.key === "KV network"));
});
test("P99 is computed from declared deterministic samples, not assigned chart coordinates", () => {
  const p = { context: 8192, base: 20, prefillRate: 80000, chunk: 1024, residual: 2 };
  const r = M.pdSamples(p);
  for (const [key, values] of Object.entries(r.samples)) {
    assert.equal(values.length, 6200);
    assert.equal(r.p99[key], [...values].sort((a, b) => a - b)[6137]);
    assert.equal(r.requestTPOT[key].length, 200);
    assert.equal(r.p99TPOT[key], [...r.requestTPOT[key]].sort((a, b) => a - b)[197]);
    assert.ok(r.p99TPOT[key] <= r.p99[key]);
  }
  assert.ok(r.p99.colocated > r.p99.chunked);
  assert.ok(r.p99.chunked > r.p99.pd);
  assert.deepEqual(r, M.pdSamples(p));
});
test("nearest-rank percentile rejects absent/non-finite samples", () => {
  assert.throws(() => M.percentile([], 0.99), RangeError);
  assert.throws(() => M.percentile([1, NaN], 0.99), RangeError);
  assert.equal(M.percentile([3, 1, 2], 0.5), 2);
});
test("NCCL bandwidth factors match total-array scopes and decimal units", () => {
  const p = { ranks: 8, sizeMB: 1000, timeMs: 20 };
  const r = M.collective({ ...p, operation: "allreduce" });
  assert.equal(r.algbw, 50);
  assert.equal(r.busbw, 87.5);
  for (const operation of ["allgather", "reducescatter", "alltoall"]) assert.equal(M.collective({ ...p, operation }).busbw, 43.75);
  assert.equal(M.collective({ ...p, operation: "sendrecv" }).busbw, 50);
  assert.throws(() => M.collective({ ...p, ranks: 1, operation: "allreduce" }), RangeError);
  assert.throws(() => M.collective({ ...p, operation: "unknown" }), RangeError);
  for (const operation of ["allgather", "reducescatter", "alltoall"]) {
    const api = M.collective({ ...p, operation, sizeMode: "api-count" });
    assert.equal(api.effectiveSizeMB, 8000);
    assert.equal(api.algbw, 400);
    assert.equal(api.busbw, 350);
  }
  assert.equal(M.collective({ ...p, operation: "allreduce", sizeMode: "api-count" }).algbw, 50);
});
test("PCIe Gen3/4/5 encoding is one-way and must not be reused for PAM4/FLIT generations", () => {
  near(M.pcieBandwidth(4, 16, 1).encodedGBps, 16 * 16 * 128 / 130 / 8);
  near(M.pcieBandwidth(5, 16, 0.85).payloadGBps, 53.56307692307692);
  assert.throws(() => M.pcieBandwidth(6, 16, 1), RangeError);
});
const network = { gpu: "h100", gpuEndpoints: 8, nicPorts: 8, nicCards: 8, rate: 400, gpuGen: 5, gpuLanes: 16, nicGen: 5, nicLanes: 16, networkEfficiency: 0.8, pcieEfficiency: 0.85, crossesShared: true, sharedGBps: 300, fabricGBps: 240, payloadMB: 1024 };
test("node bandwidth budgets distinguish ports, PCIe cards, participating GPUs and crossed shared links", () => {
  const r = M.networkBudget(network);
  assert.equal(r.nodeGBps, 240);
  near(r.stripedTransferMs, 1024 / 240);
  assert.equal(r.nvlinkOneWayGBps, 450);
  near(M.networkBudget({ ...network, nicCards: 4 }).nodeGBps, 4 * 53.56307692307692);
  near(M.networkBudget({ ...network, gpuEndpoints: 1 }).nodeGBps, 53.56307692307692);
  assert.equal(M.networkBudget({ ...network, sharedGBps: 100 }).nodeGBps, 100);
  assert.equal(M.networkBudget({ ...network, crossesShared: false, sharedGBps: undefined }).nodeGBps, 240);
  assert.equal(M.networkBudget({ ...network, gpu: "l40s" }).nvlinkOneWayGBps, null);
});
const rails = { gpus: 8, railCount: 4, payloadGB: 1, rate: 400, efficiency: 0.8, fabricGBps: 300, hotGpu: 0, hotMultiplier: 1, mapping: "balanced", failure: false, failedRail: 0, reroute: true };
test("rail mapping exposes collisions, hotspots and incomplete failed paths rather than N times line rate", () => {
  const r = M.railBudget(rails);
  assert.equal(r.lowerBoundMs, 50);
  assert.equal(r.effectiveGBps, 160);
  assert.equal(M.railBudget({ ...rails, mapping: "skewed" }).lowerBoundMs, 200);
  assert.equal(M.railBudget({ ...rails, hotMultiplier: 4 }).lowerBoundMs, 125);
  const failure = M.railBudget({ ...rails, failure: true, reroute: false });
  assert.equal(failure.blockedGB, 2);
  assert.equal(failure.lowerBoundMs, null);
  assert.equal(failure.effectiveGBps, null);
  assert.equal(M.railBudget({ ...rails, failure: true }).lowerBoundMs, 75);
  assert.equal(M.railBudget({ ...rails, fabricGBps: 100 }).lowerBoundMs, 80);
  assert.equal(M.railBudget({ ...rails, railCount: 1, failure: true }).lowerBoundMs, null);
});
test("Goodput is gated independently of unbounded throughput with declared sample denominator", () => {
  const p = { batch: 32, base: 12, slope: 0.25, knee: 48, ttftLimit: 800, tpotLimit: 35 };
  const r = M.goodput(p);
  assert.equal(r.requests.length, 32);
  assert.equal(r.passed, 32);
  assert.equal(r.goodput, r.throughput);
  const failure = M.goodput({ ...p, batch: 128 });
  assert.ok(failure.throughput > 0);
  assert.equal(failure.goodput, 0);
});
test("zero output goodput has undefined unit cost instead of free service", () => {
  const p = { hourly: 40, outputTPS: 5000, inputTPS: 20000, hours: 8, priceIn: 1, priceOut: 5 };
  near(M.cost(p).perMillionOutput, 40e6 / (3600 * 5000));
  assert.equal(M.cost(p).periodCost, 1296);
  assert.equal(M.cost({ ...p, outputTPS: 0 }).perMillionOutput, null);
});
test("PFC hysteresis resumes at Xon, ECN marks earlier and invalid thresholds are rejected", () => {
  const p = { queue: 85, ecn: 40, pfc: 80, xon: 60 };
  assert.equal(M.congestion(p).paused, true);
  assert.equal(M.congestion({ ...p, queue: 70 }, true).paused, true);
  assert.equal(M.congestion({ ...p, queue: 60 }, true).paused, false);
  assert.equal(M.congestion({ ...p, queue: 50 }).marked, true);
  assert.throws(() => M.congestion({ ...p, ecn: 90 }), RangeError);
  assert.throws(() => M.congestion({ ...p, xon: 90 }), RangeError);
  const budget = M.pfcBudget({ bufferMB: 16, rate: 400, reactionUs: 10, frameBytes: 9000, guardMB: 0.05, pfc: 80 });
  near(budget.requiredMB, 0.559);
  near(budget.availableMB, 3.2);
  assert.equal(budget.fits, true);
  assert.equal(M.pfcBudget({ bufferMB: 16, rate: 800, reactionUs: 100, frameBytes: 9000, guardMB: 0.05, pfc: 80 }).fits, false);
});
test("non-finite, negative and fractional request-count inputs fail explicitly", () => {
  assert.throws(() => M.capacity({ ...capacity, qps: NaN }), RangeError);
  assert.throws(() => M.capacity({ ...capacity, prompt: -1 }), RangeError);
  assert.throws(() => M.capacity({ ...capacity, output: 1.2 }), RangeError);
  assert.throws(() => M.lifecycle({ queue: 0, prefill: 1, delivery: 0, output: 0, tpot: 20 }), RangeError);
});
test("balanced forward PP schedules each microbatch serially without treating more batches as lower request latency", () => {
  const r = M.pipeline({ ranks: 4, batch: 8 });
  assert.equal(r.slots, 11);
  near(r.utilization, 8 / 11);
  assert.equal(r.latencySlots, 4);
  assert.equal(r.idleSlots, 12);
  assert.deepEqual(r.schedule.map((row) => row[0].start), [0, 1, 2, 3]);
  assert.equal(r.schedule[3][7].end, 11);
  assert.equal(M.pipeline({ ranks: 4, batch: 64 }).latencySlots, 4);
});
test("speculative expected commit length uses the acceptance geometric sum and counts every cycle cost", () => {
  const p = { draftTokens: 4, acceptance: 0.8, targetMs: 20, draftMs: 2, verifyExtraMs: 1, overheadMs: 1 };
  const r = M.speculative(p);
  near(r.expectedTokens, 3.3616);
  assert.equal(r.cycleMs, 33);
  near(r.probabilities.reduce((a, v) => a + v, 0), 1);
  near(r.probabilities.reduce((a, v, i) => a + v * (i + 1), 0), r.expectedTokens);
  near(r.speedup, 20 * 3.3616 / 33);
  assert.equal(M.speculative({ ...p, acceptance: 0 }).expectedTokens, 1);
  assert.equal(M.speculative({ ...p, acceptance: 1 }).expectedTokens, 5);
  assert.ok(M.speculative({ ...p, acceptance: 0 }).speedup < 1);
  assert.throws(() => M.speculative({ ...p, acceptance: 1.1 }), RangeError);
});
test("prefix request hit rate is not the fraction of all input avoided; misses and lookup can hurt", () => {
  const p = { model: "llama70", kvBytes: 2, prompt: 8192, prefix: 4096, hitRate: 0.7, prefillRate: 20000, lookupMs: 2, entries: 10 };
  const r = M.prefixCaching(p);
  near(r.avoidedTokens, 2867.2);
  near(r.inputComputeTokens, 5324.8);
  near(r.baselineMs, 409.6);
  near(r.expectedMs, 268.24);
  near(r.cacheGB, 13.4217728);
  assert.equal(M.prefixCaching({ ...p, hitRate: 0 }).savedMs, -2);
  assert.equal(M.prefixCaching({ ...p, prefix: 0 }).cacheGB, 0);
  assert.throws(() => M.prefixCaching({ ...p, prefix: 8193 }), RangeError);
});
test("publication generation rejects missing citations, private URLs, missing translations and false hardware provenance", () => {
  const source = JSON.parse(readFileSync(join(directory,"sources.json"),"utf8"));
  assert.doesNotThrow(()=>validateContentSources(source));
  const missing=structuredClone(source);
  missing.sources=missing.sources.filter((s)=>s.id!=="h100");
  assert.throws(()=>validateContentSources(missing));
  const privateURL=structuredClone(source);
  for (const url of ["https://localhost/reference","https://172.16.0.1/reference","https://[::1]/reference","https://reader:password@example.com/reference"]) {
    privateURL.sources[0].url=url;
    assert.throws(()=>validateContentSources(privateURL));
  }
  const duplicate=structuredClone(source);
  duplicate.sources.push(duplicate.sources[0]);
  assert.throws(()=>validateContentSources(duplicate));
  assert.throws(()=>validateContentSources({...source,reviewed_at:"2026-02-30"}));
  assert.throws(()=>validateContentSources({...source,hardware_measured:true}));
  assert.throws(()=>b("只有中文",""));
  const zh=readFileSync(join(directory,"..","inference-sizing-network","faq.md"),"utf8").match(/^### Q\d+\./gm);
  const en=readFileSync(join(directory,"..","inference-sizing-network","faq.en.md"),"utf8").match(/^### Q\d+\./gm);
  const expected=Array.from({length:24},(_,i)=>`### Q${i+1}.`);
  assert.deepEqual(zh,expected);
  assert.deepEqual(en,expected);
});
test("all seven pages are standalone, CSP hashes seal actual scripts, bilingual structure and local navigation are present", () => {
  const files = readdirSync(directory).filter((f) => f.endsWith(".html"));
  assert.equal(files.length, 7);
  for (const filename of files) {
    const html = readFileSync(join(directory, filename), "utf8");
    assert.doesNotMatch(html, /<script[^>]+src=|<link[^>]+(?:stylesheet|preload)|<iframe|<img[^>]+src=["']https?:/i);
    assert.doesNotMatch(html, /@import|url\(["']?https?:|fetch\(|XMLHttpRequest/);
    assert.match(html, /connect-src &#39;none&#39;|connect-src 'none'/);
    assert.match(html, /data-lang="zh-CN"/);
    assert.match(html, /data-lang="en"/);
    const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
    assert.equal(scripts.length, 3);
    for (const script of scripts) {
      new Script(script);
      const hash = createHash("sha256").update(script).digest("base64");
      assert.ok(html.includes("sha256-" + hash), `${filename}: script hash missing`);
    }
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
    assert.equal(new Set(ids).size, ids.length, `${filename}: duplicate ids`);
    for (const [, href] of html.matchAll(/\bhref="([^"]+)"/g)) {
      if (href.startsWith("#")) {
        assert.ok(ids.includes(href.slice(1)), `${filename}: missing anchor ${href}`);
      } else if (!/^https?:/.test(href)) {
        const path = decodeURIComponent(href.split(/[?#]/)[0]);
        assert.doesNotThrow(() => readFileSync(resolve(directory, path)), `${filename}: missing link ${path}`);
      }
    }
  }
});
