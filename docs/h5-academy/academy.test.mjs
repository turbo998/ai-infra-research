import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { Script } from "node:vm";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import M from "./academy-math.cjs";

const directory = dirname(fileURLToPath(import.meta.url));
const near = (actual, expected, epsilon = 1e-7) => assert.ok(Math.abs(actual - expected) < epsilon, `${actual} != ${expected}`);
const sizing = { model: "llama70", gpu: "h200", weightBytes: 1, kvBytes: 2, batch: 32, context: 8192, prefillBatch: 1, safe: 0.85, runtime: 3, overhead: 0.05, replication: 1, mbu: 0.6, mfu: 0.4, tpot: 20, ttft: 800, reserve: 100, mtp: false };
const capacity = { qps: 10, headroom: 1.3, prompt: 2048, output: 512, ttft: 800, tpot: 20, inputCapacity: 20000, outputCapacity: 1500, batch: 128, utilization: 0.7 };
const transfer = { model: "llama70", context: 8192, kvBytes: 1, cache: 0, rate: 400, efficiency: 0.8, rails: 1, setup: 0.2, overlap: 10, qps: 10 };

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
test("P99 is computed from declared deterministic samples, not assigned chart coordinates", () => {
  const p = { context: 8192, base: 20, prefillRate: 80000, chunk: 1024, residual: 2 };
  const r = M.pdSamples(p);
  for (const [key, values] of Object.entries(r.samples)) {
    assert.equal(values.length, 200);
    assert.equal(r.p99[key], [...values].sort((a, b) => a - b)[197]);
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
});
test("non-finite, negative and fractional request-count inputs fail explicitly", () => {
  assert.throws(() => M.capacity({ ...capacity, qps: NaN }), RangeError);
  assert.throws(() => M.capacity({ ...capacity, prompt: -1 }), RangeError);
  assert.throws(() => M.capacity({ ...capacity, output: 1.2 }), RangeError);
  assert.throws(() => M.lifecycle({ queue: 0, prefill: 1, delivery: 0, output: 0, tpot: 20 }), RangeError);
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
